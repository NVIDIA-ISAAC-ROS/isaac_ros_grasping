#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Download GraspGenX Hugging Face checkpoints and export ONNX for Isaac ROS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

MODEL_REPO = 'adithyamurali/GraspGenXModel'
MODEL_REVISION = '7c834043c11a11417e31d6d5ea9355801e40a2c1'
CHECKPOINT_FILES = (
    'release/gen/config.yaml', 'release/gen/epoch_736.pth',
    'release/dis/config.yaml', 'release/dis/epoch_1056.pth',
)
INPUT_NAMES = ['feat', 'coord', 'grid_coord', 'code', 'order', 'inverse',
               'sweep_volume', 'init_noise', 'step_noise']


# Replace redundant Where(condition, x, x) nodes with Identity when static shapes match.
def eliminate_where_nodes(model):
    shapes = {value.name: tuple(dim.dim_value if dim.HasField('dim_value') else None
                                for dim in value.type.tensor_type.shape.dim)
              for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
              if value.type.tensor_type.HasField('shape')}
    shapes.update({value.name: tuple(value.dims) for value in model.graph.initializer})
    for node in model.graph.node:
        if node.domain or node.op_type != 'Where' or node.input[1] != node.input[2]:
            continue
        shape = shapes.get(node.input[1])
        if shape is None or None in shape or shapes.get(node.output[0]) != shape:
            continue
        branch = node.input[1]
        node.op_type = 'Identity'
        del node.input[:]
        node.input.append(branch)
        del node.attribute[:]


def export(checkpoints, output):
    # The release requires the native C++ plugin operators, not Python QDP.
    os.environ['GRASPGENX_USE_CPP_PLUGINS'] = '1'
    script_dir = Path(__file__).resolve().parent
    vendor = script_dir.parent / 'vendor'
    if not vendor.is_dir():
        vendor = script_dir.parents[1] / 'share' / 'isaac_ros_graspgenx_models_install' / 'vendor'
    sys.path.insert(0, str(vendor))

    import onnx
    from onnxscript import FLOAT, INT64
    from onnxscript.values import Opset
    import torch

    from graspgenx.models.grasp_gen import GraspGen
    from graspgenx.model_opt.everything_graph import build_everything_graph, everything_inputs
    from graspgenx.utils.checkpoint_io import load_model_cfg

    cfg = load_model_cfg(
        checkpoints / 'release/gen', checkpoints / 'release/dis',
        gen_pth='epoch_736.pth', dis_pth='epoch_1056.pth')
    torch.manual_seed(8031)
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    for network, filename in ((model.grasp_generator, cfg.eval.gen_checkpoint),
                              (model.grasp_discriminator, cfg.eval.dis_checkpoint)):
        state = torch.load(filename, map_location='cpu', weights_only=True)
        network.load_state_dict(state['model'])
    model = model.cuda().eval()
    module = build_everything_graph(model, 3500, 200, trt_plugins=True)
    module.gen_encoder.fp32_attn = True
    module.dis_encoder.fp32_attn = True
    prep, init, step, _ = everything_inputs(model, torch.randn(3500, 3, device='cuda') * .05, 200)
    example = (*prep, torch.randn(1, 12, device='cuda'), init, step)

    plugins = Opset('graspgenx', 1)
    # ONNX Script needs schemas even though TensorRT implements these operators.
    for name, arity in [('gather0', 2), ('scatter_reduce', 3),
                        ('hash_conv', 6), ('pool_cluster', 3)]:
        schema = onnx.defs.OpSchema
        attributes = [schema.Attribute(key, schema.AttrType.STRING, key)
                      for key in ('plugin_namespace', 'plugin_version')]
        if name == 'scatter_reduce':
            attributes.append(schema.Attribute('reduce', schema.AttrType.STRING, 'Reduction'))
        onnx.defs.register_schema(schema(
            name, 'graspgenx', 1,
            inputs=[schema.FormalParameter(f'input{i}', f'T{i}') for i in range(arity)],
            outputs=[schema.FormalParameter('output', 'Tout')],
            type_constraints=[(typ, ['tensor(float)', 'tensor(int32)', 'tensor(int64)'], typ)
                              for typ in [*(f'T{i}' for i in range(arity)), 'Tout']],
            attributes=attributes))

    def gather0(src: FLOAT, index: INT64) -> FLOAT:
        return plugins.gather0(src, index, plugin_namespace='graspgenx', plugin_version='1')

    def scatter_reduce(dst: FLOAT, index: INT64, src: FLOAT, reduce: str) -> FLOAT:
        return plugins.scatter_reduce(dst, index, src, reduce=reduce,
                                      plugin_namespace='graspgenx', plugin_version='1')

    def hash_conv(feat: FLOAT, grid_coord: INT64, batch: INT64, weight: FLOAT,
                  bias: FLOAT, offsets: INT64) -> FLOAT:
        return plugins.hash_conv(feat, grid_coord, batch, weight, bias, offsets,
                                 plugin_namespace='graspgenx', plugin_version='1')

    def pool_cluster(c0: INT64, code_shifted: INT64, n: INT64) -> INT64:
        return plugins.pool_cluster(c0, code_shifted, n,
                                    plugin_namespace='graspgenx', plugin_version='1')

    translations = {
        torch.ops.graspgenx.gather0.default: gather0,
        torch.ops.graspgenx.scatter_reduce.default: scatter_reduce,
        torch.ops.graspgenx.hash_conv.default: hash_conv,
        torch.ops.graspgenx.pool_cluster.default: pool_cluster,
    }
    with torch.inference_mode():
        program = torch.onnx.export(
            module, example, dynamo=True, optimize=True, opset_version=18,
            input_names=INPUT_NAMES, output_names=['output0', 'output1'],
            custom_translation_table=translations)
    output.mkdir(parents=True, exist_ok=True)
    graph = output / 'graspgenx_everything.onnx'
    program.save(str(graph), external_data=True)
    graph_model = onnx.load(str(graph), load_external_data=False)
    eliminate_where_nodes(graph_model)
    onnx.save_model(graph_model, str(graph))

    # The existing engine installer requires checksums for these two files.
    files = {}
    for name in ('graspgenx_everything.onnx', 'graspgenx_everything.onnx.data'):
        with (output / name).open('rb') as stream:
            files[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (output / 'model_manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    checkpoints = args.checkpoint_dir.resolve()
    for name in CHECKPOINT_FILES:
        target = checkpoints / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
        urllib.request.urlretrieve(url, target)
    export(checkpoints, args.output_dir.resolve())


if __name__ == '__main__':
    main()
