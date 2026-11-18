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

"""The whole GraspGenX inference as ONE traceable graph.

Composes the export-friendly PTv3 encoders (graspgenx/models/ptv3/
ptv3_vanilla_export.py) with the fused rollout+scoring module
(graspgenx/models/ddpm_rollout.py):

    raw points (+ host-side serialization codes) + gripper params + noise
        -> generator encoder  ┐
        -> discriminator enc  ┼-> unrolled DDPM rollout -> pose conversion
        -> gripper MLPs       ┘   -> discriminator scoring
        -> grasps [K, 4, 4], confidence [K, 1]

Host-side prep (cheap, integer/prep work analogous to feeding the noise):
kappa-scaling + centering of the point cloud, Morton/Hilbert serialization
(``ptv3_export_inputs``), and the Gaussian noise draws.

One object with a fixed point budget N and fixed grasp count K per call.
See specs/ptv3_export.md.

With ``trt_plugins=True`` (see graspgenx/models/trt_plugins.py) the WHOLE
module — encoders included — compiles to a single TensorRT engine via
``require_full_compilation=True`` (specs/trt_plugins.md). Without it, the
encoders' data-dependent ops keep the module as a torch.export / ONNX
artifact only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graspgenx.model_opt.ddpm_rollout import (
    UnrolledRolloutWithScoring,
    build_fused_rollout_scoring,
)
from graspgenx.model_opt.ptv3_vanilla_export import (
    PTv3VanillaExport,
    ptv3_export_inputs,
)


class GraspGenXEverything(nn.Module):
    """forward(feat [N,3], coord [N,3], grid_coord [N,3], code [O,N],
    order [O,N], inverse [O,N], sweep_volume [1,12], init_noise [K,6],
    step_noise [T,K,6]) -> (grasps [K,4,4], confidence [K,1])."""

    def __init__(
        self,
        gen_encoder: PTv3VanillaExport,
        dis_encoder: PTv3VanillaExport,
        gen_gripper: nn.Module,
        dis_gripper: nn.Module,
        fused: UnrolledRolloutWithScoring,
        num_grasps: int,
    ):
        super().__init__()
        self.gen_encoder = gen_encoder
        self.dis_encoder = dis_encoder
        self.gen_gripper = gen_gripper
        self.dis_gripper = dis_gripper
        self.fused = fused
        self.num_grasps = num_grasps

    @property
    def num_steps(self) -> int:
        return self.fused.num_steps

    def forward(
        self, feat, coord, grid_coord, code, order, inverse,
        sweep_volume, init_noise, step_noise,
    ):
        # K is derived from the noise so the same graph serves any grasp count
        # (a TRT dynamic-shape profile on init_noise/step_noise makes one engine
        # sample variable K); for the static build it just equals num_grasps.
        K = init_noise.shape[0]
        gen_obj = self.gen_encoder(feat, coord, grid_coord, code, order, inverse)
        dis_obj = self.dis_encoder(feat, coord, grid_coord, code, order, inverse)
        gen_obs = torch.cat([gen_obj, self.gen_gripper(sweep_volume)], dim=-1)
        dis_obs = torch.cat([dis_obj, self.dis_gripper(sweep_volume)], dim=-1)
        return self.fused(
            gen_obs.expand(K, -1), dis_obs.expand(K, -1), init_noise, step_noise
        )


def build_everything_graph(
    model,
    num_points: int,
    num_grasps: int,
    trt_plugins: bool = False,
    reduce_capacity: bool = None,
) -> GraspGenXEverything:
    """Build the everything-graph from a loaded GraspGen model.

    Requires the release-style configuration: both encoders ptv3vanilla with
    IDENTICAL preprocessing (same kappa and grid_size — so one set of point /
    serialization inputs feeds both) and sweep_volume_v2 gripper conditioning.

    trt_plugins=True routes the converter-less ops (scatter-reduce, 1-D
    argsort) through the graspgenx TensorRT QDP plugins for a pure
    single-engine TRT build (numerically identical; see
    specs/trt_plugins.md).

    reduce_capacity shrinks the deep encoder stages' buffers to nearly halve
    encoder latency (see reduced_capacity_schedule) at ~3e-3 embedding cost.
    Defaults to ``trt_plugins`` (on for the TRT engine, off for the exact
    ONNX / torch.export path).
    """
    from graspgenx.model_opt.ptv3_vanilla_export import (
        reduced_capacity_schedule,
    )

    if reduce_capacity is None:
        reduce_capacity = trt_plugins
    caps = reduced_capacity_schedule(num_points) if reduce_capacity else None
    gen, dis = model.grasp_generator, model.grasp_discriminator
    for side in (gen, dis):
        if side.object_backbone != "ptv3vanilla":
            raise ValueError(f"object_backbone must be ptv3vanilla, got {side.object_backbone!r}")
        if side.gripper_backbone != "sweep_volume_v2":
            raise ValueError(
                f"gripper_backbone must be sweep_volume_v2, got {side.gripper_backbone!r}"
            )
    if gen.kappa != dis.kappa or gen.grid_size != dis.grid_size:
        raise ValueError(
            "generator/discriminator preprocessing differs "
            f"(kappa {gen.kappa} vs {dis.kappa}, grid {gen.grid_size} vs "
            f"{dis.grid_size}); a shared point/serialization input requires "
            "identical preprocessing."
        )

    fused = build_fused_rollout_scoring(model)
    module = GraspGenXEverything(
        PTv3VanillaExport(
            gen.object_encoder, num_points, plugin_ops=trt_plugins, capacities=caps
        ),
        PTv3VanillaExport(
            dis.object_encoder, num_points, plugin_ops=trt_plugins, capacities=caps
        ),
        getattr(gen.gripper_encoder, "eager_module", gen.gripper_encoder),
        getattr(dis.gripper_encoder, "eager_module", dis.gripper_encoder),
        fused,
        num_grasps,
    )
    device = next(gen.parameters()).device
    return module.to(device).eval()


def everything_inputs(model, obj_pc: torch.Tensor, num_grasps: int, generator=None):
    """Host-side prep for one object: center + kappa-scale the PC, run the
    eager serialization, draw noise. Returns (args tuple, center [3])."""
    gen = model.grasp_generator
    device = next(gen.parameters()).device
    pc = obj_pc.to(device=device, dtype=torch.float32)
    center = pc.mean(dim=0)
    scaled = (pc - center) * (gen.kappa if gen.kappa is not None else 1.0)

    prep = ptv3_export_inputs(scaled, gen.grid_size, gen.object_encoder.order)
    T = gen.num_diffusion_iters_eval
    init = torch.randn(num_grasps, gen.output_dim, device=device, generator=generator)
    step = torch.randn(T, num_grasps, gen.output_dim, device=device, generator=generator)
    args = (
        prep["feat"], prep["coord"], prep["grid_coord"],
        prep["code"], prep["order"], prep["inverse"],
    )
    return args, init, step, center
