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

"""TensorRT QDP plugins for the ops torch_tensorrt cannot convert.

Two Python-defined TensorRT plugins (`tensorrt.plugin`, TRT >= 10.6) close
the converter gaps that block a pure single-engine build of the export
encoder (specs/trt_plugins.md):

- ``graspgenx::scatter_reduce`` — functional scatter-reduce over dim 0 with
  include_self=True (exactly the export module's pooling usage).
- ``graspgenx::argsort`` — 1-D argsort for any dtype including int64 (TRT's
  TopK caps K at 3840 and the hash-lookup sort is ~98k elements).

Both are torch custom ops whose EAGER implementation is the plain torch op,
so routing through them changes nothing numerically outside TRT; inside TRT
the QDP ``impl`` runs the same torch CUDA kernels on zero-copy views of the
engine's buffers, on the engine's stream.

Importing this module registers the custom ops and the QDP plugins. Call
:func:`register_trt_plugin_converters` once before ``torch_tensorrt`` compile
to install the auto-generated converters.

PORTABILITY: engines built with these plugins load only in a Python process
that has imported this module first (QDP plugins are Python-defined) — not
in pure C++ TensorRT. See specs/trt_plugins.md.
"""

# NOTE: no `from __future__ import annotations` here — tensorrt.plugin's
# register decorator inspects live annotation classes, not strings.
from typing import List, Tuple

import torch

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── torch custom ops (eager impl == the plain torch op) ─────────────────────


@torch.library.custom_op("graspgenx::scatter_reduce", mutates_args=())
def _scatter_reduce_op(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor, reduce: str
) -> torch.Tensor:
    return torch.scatter_reduce(dst, 0, index, src, reduce=reduce, include_self=True)


@_scatter_reduce_op.register_fake
def _(dst, index, src, reduce):
    return torch.empty_like(dst)


@torch.library.custom_op("graspgenx::argsort", mutates_args=())
def _argsort_op(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(x)


@_argsort_op.register_fake
def _(x):
    return torch.empty(x.shape, dtype=torch.long, device=x.device)


@torch.library.custom_op("graspgenx::gather0", mutates_args=())
def _gather0_op(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # src[index] along dim 0 (advanced indexing by a 1-D int index).
    return src[index]


@_gather0_op.register_fake
def _(src, index):
    return torch.empty(
        (index.shape[0], *src.shape[1:]), dtype=src.dtype, device=src.device
    )


# Coarse plugin: the submanifold hash-conv neighbor lookup + gather + weighted
# sum, as ONE op. Encapsulates all the int64 glue (sort / searchsorted-via-
# cumsum / eq mask / index gathers) that torch_tensorrt 2.6 mis-converts, so
# the TRT graph never sees those ops. Impl mirrors
# PTv3VanillaExport._hash_conv exactly; weights are passed as inputs so a
# single plugin serves every conv instance.
_HASH_P = (73856093, 19349669, 83492791, 334214467)


def _hash_keys(batch, grid_coord):
    P = _HASH_P
    return (
        batch.long() * P[3]
        + grid_coord[:, 0].long() * P[0]
        + grid_coord[:, 1].long() * P[1]
        + grid_coord[:, 2].long() * P[2]
    )


def _hash_conv_impl(feat, grid_coord, batch, weight, bias, offsets):
    # Mirrors PTv3VanillaExport._hash_conv (the cat-sort + cumsum lookup) so
    # the plugin path is bit-identical to the native path. bias is always a
    # C_out tensor (zeros when the conv is bias-free) to avoid a 0-element
    # TRT constant.
    N = feat.shape[0]
    K3 = offsets.shape[0]
    device = feat.device
    keys = _hash_keys(batch, grid_coord)
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    neighbor_coords = grid_coord.unsqueeze(1) + offsets.unsqueeze(0)
    neighbor_batch = batch.unsqueeze(1).expand(-1, K3)
    neighbor_keys = _hash_keys(neighbor_batch.reshape(-1), neighbor_coords.reshape(-1, 3))
    Q = neighbor_keys.shape[0]
    merged = torch.cat([sorted_keys * 2, neighbor_keys * 2 + 1])
    morder = torch.argsort(merged)
    is_table = (morder < N).long()
    cnt = torch.cumsum(is_table, dim=0) - 1
    inv_perm = torch.zeros_like(morder)
    inv_perm.scatter_(0, morder, torch.arange(N + Q, device=device))
    q_pos = inv_perm[N:]
    positions = cnt[q_pos].clamp(min=0, max=N - 1)
    found_mask = (cnt[q_pos] >= 0) & (sorted_keys[positions] == neighbor_keys)
    found_indices = sort_idx[positions]
    neighbor_feat = feat[found_indices] * found_mask.unsqueeze(-1)
    neighbor_feat = neighbor_feat.view(N, K3, feat.shape[1])
    out = torch.einsum("nki,kio->no", neighbor_feat, weight)
    return out + bias


_SENTINEL_BASE = 1 << 30  # int32-safe; must match ptv3_vanilla_export._SENTINEL_BASE


def _pool_cluster_impl(c0, code_shifted, n, num_points):
    """All integer clustering glue of one pooling stage, as one op.
    c0: [N] int64 (valid codes, invalid=sentinel), code_shifted: [O,N] int64,
    n: [] int64 live count. Returns sidx, cluster, n_new, head_idx, new_code,
    new_order, new_inverse. Mirrors PTv3VanillaExport._pool exactly."""
    N = num_points
    device = c0.device
    arange = torch.arange(N, device=device)
    sentinel = _SENTINEL_BASE + arange
    sidx = torch.argsort(c0)
    sorted_c = c0[sidx]
    valid_sorted = arange < n
    prev = torch.cat([sorted_c.new_full((1,), -1), sorted_c[:-1]])
    flags = (sorted_c != prev) & valid_sorted
    cluster = (torch.cumsum(flags.long(), dim=0) - 1).clamp(min=0)
    cluster = torch.where(valid_sorted, cluster, torch.full_like(cluster, N - 1))
    n_new = flags.sum()
    valid_new = arange < n_new
    head_pos = torch.full((N,), N - 1, device=device, dtype=torch.long)
    head_pos.scatter_reduce_(0, cluster, arange, reduce="amin", include_self=True)
    head_idx = sidx[head_pos.clamp(max=N - 1)]
    new_code = code_shifted[:, head_idx]
    new_code = torch.where(valid_new.unsqueeze(0), new_code, sentinel.unsqueeze(0))
    O = code_shifted.shape[0]
    new_order = torch.stack([torch.argsort(new_code[o]) for o in range(O)])
    new_inverse = torch.zeros_like(new_order).scatter_(
        dim=1, index=new_order, src=arange.unsqueeze(0).expand(O, -1)
    )
    return sidx, cluster, n_new, head_idx, new_code, new_order, new_inverse


def _pool_cluster_packed_len(N, O):
    # sidx(N) cluster(N) head_idx(N) n_new(1) new_code(O*N) new_order(O*N)
    # new_inverse(O*N)
    return 3 * N + 1 + 3 * O * N


def _pool_cluster_pack(c0, code_shifted, n, N):
    sidx, cluster, n_new, head_idx, new_code, new_order, new_inverse = (
        _pool_cluster_impl(c0, code_shifted, n, N)
    )
    return torch.cat(
        [
            sidx, cluster, head_idx, n_new.reshape(1),
            new_code.reshape(-1), new_order.reshape(-1), new_inverse.reshape(-1),
        ]
    )


@torch.library.custom_op("graspgenx::pool_cluster", mutates_args=())
def _pool_cluster_custom(
    c0: torch.Tensor, code_shifted: torch.Tensor, n: torch.Tensor
) -> torch.Tensor:
    # SINGLE packed output: torch_tensorrt 2.6's generate_plugin_converter
    # mis-handles multi-output QDP plugins (returns one ITensor, indexing
    # breaks + heap corruption). Packing into one 1-D int64 tensor sidesteps
    # it; the module unpacks with static slices + reshape (both convertible).
    return _pool_cluster_pack(c0, code_shifted, n, c0.shape[0])


@_pool_cluster_custom.register_fake
def _(c0, code_shifted, n):
    N = c0.shape[0]
    O = code_shifted.shape[0]
    return torch.empty(
        _pool_cluster_packed_len(N, O), dtype=torch.long, device=c0.device
    )


@torch.library.custom_op("graspgenx::hash_conv", mutates_args=())
def _hash_conv_custom(
    feat: torch.Tensor,
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return _hash_conv_impl(feat, grid_coord, batch, weight, bias, offsets)


@_hash_conv_custom.register_fake
def _(feat, grid_coord, batch, weight, bias, offsets):
    return torch.empty(
        (feat.shape[0], weight.shape[2]), dtype=feat.dtype, device=feat.device
    )


def _flash_attn_impl(q, k, v, bias):
    import torch.nn.functional as F

    dtype = q.dtype
    out = F.scaled_dot_product_attention(
        q.half(), k.half(), v.half(), attn_mask=bias.half()
    )
    return out.to(dtype)


@torch.library.custom_op("graspgenx::flash_attn", mutates_args=())
def _flash_attn_custom(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # fp16 flash SDPA — identical to the eager encoder's attention, so it
    # matches the true model. TRT's OWN fp16 SDPA kernel diverges badly
    # (~0.85 embedding error); running it through this plugin (torch's kernel)
    # is both correct AND fast, avoiding the 50 ms/inference fp32-attention
    # penalty. scale defaults to 1/sqrt(head_dim), matching attn.scale.
    return _flash_attn_impl(q, k, v, bias)


@_flash_attn_custom.register_fake
def _(q, k, v, bias):
    return torch.empty_like(q)


# ── TensorRT QDP plugin registrations ────────────────────────────────────────

try:
    # Skipped when GRASPGENX_USE_CPP_PLUGINS=1 — the native C++ plugins
    # (cpp/plugins/) register the same graspgenx::<op>::1 identity, so
    # registering the Python QDP creators too would collide (get_creator would
    # return the QDP one). Raising here routes to the except below, which sets
    # _QDP_AVAILABLE=False. The torch custom ops above register either way.
    import os as _os
    if _os.environ.get("GRASPGENX_USE_CPP_PLUGINS") == "1":
        raise RuntimeError("QDP disabled: using native C++ plugins")

    import tensorrt.plugin as trtp

    def _ci(t):
        # Zero-copy torch view of a plugin input's TRT buffer. (An earlier
        # defensive .clone() here was found unnecessary once the real
        # correctness bugs — int64 sentinel truncation and TRT's fp16 SDPA —
        # were fixed; dropping it removes a copy on every plugin input.)
        return torch.as_tensor(t, device="cuda")

    @trtp.register("graspgenx::scatter_reduce")
    def _(
        dst: trtp.TensorDesc, index: trtp.TensorDesc, src: trtp.TensorDesc, reduce: str
    ) -> trtp.TensorDesc:
        return dst.like()

    @trtp.impl("graspgenx::scatter_reduce")
    def _(dst, index, src, reduce, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            out_t = torch.as_tensor(outputs[0], device="cuda")
            out_t.copy_(
                torch.scatter_reduce(
                    _ci(dst), 0, _ci(index), _ci(src),
                    reduce=reduce, include_self=True,
                )
            )

    @trtp.register("graspgenx::argsort")
    def _(x: trtp.TensorDesc) -> trtp.TensorDesc:
        out = x.like()
        out.dtype = trt_int64()
        return out

    @trtp.impl("graspgenx::argsort")
    def _(x, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            out_t = torch.as_tensor(outputs[0], device="cuda")
            out_t.copy_(torch.argsort(_ci(x)))

    @trtp.register("graspgenx::gather0")
    def _(src: trtp.TensorDesc, index: trtp.TensorDesc) -> trtp.TensorDesc:
        # Output rank == src rank, dim0 == index length, dtype == src dtype.
        out = src.like()
        out.shape_expr[0] = index.shape_expr[0]
        return out

    @trtp.impl("graspgenx::gather0")
    def _(src, index, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            out_t = torch.as_tensor(outputs[0], device="cuda")
            out_t.copy_(_ci(src)[_ci(index)])

    @trtp.register("graspgenx::pool_cluster")
    def _(
        c0: trtp.TensorDesc, code_shifted: trtp.TensorDesc, n: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        N = c0.shape_expr[0]
        O = code_shifted.shape_expr[0]
        total = N * 3 + 1 + O * N * 3
        return trtp.from_shape_expr((total,), dtype=trt_int64())

    @trtp.impl("graspgenx::pool_cluster")
    def _(c0, code_shifted, n, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            N = c0.shape[0]
            packed = _pool_cluster_pack(_ci(c0), _ci(code_shifted), _ci(n), N)
            torch.as_tensor(outputs[0], device="cuda").copy_(packed)

    @trtp.register("graspgenx::flash_attn")
    def _(
        q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc,
        bias: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.impl("graspgenx::flash_attn")
    def _(q, k, v, bias, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            torch.as_tensor(outputs[0], device="cuda").copy_(
                _flash_attn_impl(_ci(q), _ci(k), _ci(v), _ci(bias))
            )

    @trtp.register("graspgenx::hash_conv")
    def _(
        feat: trtp.TensorDesc,
        grid_coord: trtp.TensorDesc,
        batch: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        bias: trtp.TensorDesc,
        offsets: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        out = feat.like()
        out.shape_expr[1] = weight.shape_expr[2]  # C_out
        return out

    @trtp.impl("graspgenx::hash_conv")
    def _(feat, grid_coord, batch, weight, bias, offsets, outputs, stream):
        s = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(s):
            out_t = torch.as_tensor(outputs[0], device="cuda")
            out_t.copy_(
                _hash_conv_impl(
                    _ci(feat), _ci(grid_coord), _ci(batch),
                    _ci(weight), _ci(bias), _ci(offsets),
                )
            )

    _QDP_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on TRT version
    logger.warning(f"tensorrt.plugin (QDP) unavailable ({e}); TRT plugin path disabled.")
    _QDP_AVAILABLE = False


def trt_int64():
    import tensorrt as trt

    return trt.DataType.INT64


_CONVERTERS_REGISTERED = False


def register_trt_plugin_converters() -> bool:
    """Install torch_tensorrt converters for the graspgenx plugins.

    Idempotent. Returns True if the converters are available."""
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return True
    if not _QDP_AVAILABLE:
        return False
    from torch_tensorrt.dynamo.conversion.plugins import generate_plugin_converter

    for op in (
        "scatter_reduce", "argsort", "gather0", "hash_conv", "pool_cluster",
        "flash_attn",
    ):
        generate_plugin_converter(f"graspgenx::{op}", supports_dynamic_shapes=False)
    _CONVERTERS_REGISTERED = True
    logger.info(
        "Registered TRT plugin converters: scatter_reduce, argsort, gather0, hash_conv"
    )
    return True
