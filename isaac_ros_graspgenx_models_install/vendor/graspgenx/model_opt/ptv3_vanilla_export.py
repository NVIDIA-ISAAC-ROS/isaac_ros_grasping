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

"""Export-friendly forward path for PointTransformerV3Vanilla.

Same weights, same math, static shapes — see specs/ptv3_export.md.

The eager encoder cannot be traced (data-dependent `torch.unique` pooling,
`.tolist()` patch sizing, Python padding loops, addict containers, randperm
order shuffling). This module re-expresses the SAME computation over
fixed-capacity buffers:

- Serialization (Morton/Hilbert codes, orders, inverses) is computed on the
  HOST by the unmodified eager code path and passed as engine inputs
  (`ptv3_export_inputs`). The graph only ever right-shifts those codes.
- Every stage keeps capacity N (= the input point budget) with a live-count
  scalar tensor `n`; invalid rows are zeroed, invalid codes carry per-slot
  sentinels that sort to the end and hash-collide with nothing.
- `unique`-based pooling becomes sort -> boundary flags -> cumsum cluster ids
  -> scatter_reduce (amax feat / mean coord), matching segment_csr_vanilla.
- Serialized attention uses the static patch size K with the eager module's
  two regimes made explicit: duplication-padding indices when n > K (the
  duplicated keys really participate, as in eager) and additive key masking
  when n <= K (eager runs a single unpadded patch of n tokens).
- One object per forward (batch handling collapses the eager padding loop).
- Order shuffling is pinned to identity (a fixed sample of the eager
  random behavior; parity is defined against eager with pinned shuffle).

Exact bitwise parity is not expected (sort tie-breaks among points sharing a
voxel, padded fp16 SDPA reduction widths); differences must stay far below
the eager encoder's own shuffle-induced run-to-run drift (~3e-2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graspgenx.models.ptv3.ptv3_vanilla import (
    PointTransformerV3Vanilla,
    VanillaBlock,
    VanillaEmbedding,
    VanillaPoint,
    VanillaSerializedPooling,
)

_NEG_MASK = -30000.0  # exp() underflows to exactly 0 in fp16/fp32 softmax
# int32-SAFE sentinel: TensorRT truncates int64 -> int32 for these code
# tensors, so the sentinel must fit in int32 while still exceeding every real
# serialization code. Release codes are < 2^24 (depth<=8, 3 dims, batch=0);
# 2^30 sits well above them and below 2^31. Must match trt_plugins._SENTINEL_BASE.
_SENTINEL_BASE = 1 << 30
_FAR_GRID = 1 << 24  # invalid grid-coord sentinel (>> real coords, int32-safe)


def ptv3_export_inputs(scaled_pc: torch.Tensor, grid_size: float, orders):
    """Host-side prep: exact eager preprocessing + serialization.

    Args:
        scaled_pc: [N, 3] centered, kappa-scaled point cloud (float32).
        grid_size: encoder grid size.
        orders: the encoder's serialization orders (model.order).
    Returns:
        dict of engine inputs: feat [N,3], coord [N,3], grid_coord [N,3]
        (int32), code [O,N] (int64), order [O,N] (int64), inverse [O,N]
        (int64); plus 'depth' (python int, for the pooling-depth guard).
    """
    from graspgenx.models.model_utils import convert_to_ptv3_pc_format

    data = convert_to_ptv3_pc_format(scaled_pc[None], grid_size=grid_size)
    point = VanillaPoint(data)
    point.serialization(order=orders, shuffle_orders=False)
    return {
        "feat": point.feat,
        "coord": point.coord,
        "grid_coord": point.grid_coord,
        "code": point.serialized_code,
        "order": point.serialized_order,
        "inverse": point.serialized_inverse,
        "depth": int(point.serialized_depth),
    }


def reduced_capacity_schedule(num_points: int, num_stages: int = 5):
    """Per-stage capacity schedule that shrinks the deep stages' buffers to
    cut padded compute (see PTv3VanillaExport). Sized for the release model
    with headroom over the measured worst-case pooling (a stage's buffer is
    kept >= the largest pooled count observed across varied inputs). Early
    stages barely pool for spread-out clouds so they keep near-full capacity;
    the wins are the deep stages (enc3 ~2.3x, enc4 ~6.8x smaller).

    Scales the 3500-tuned schedule proportionally for other point budgets.

    Real segmented object point clouds from the demo scenes pool to at most
    ~[1320, 589, 197, 59] across 16 objects, so this schedule keeps large
    headroom (the deep stages carry <200 real points into 1536/512 slots).
    The deep-stage reductions capture nearly all the speedup — going more
    aggressive on the buffer sizes barely helps (per-plugin-call overhead is
    the floor, not tensor size) while costing accuracy, so headroom is kept.
    A pathologically spread cloud that pools less has excess points dropped
    (not a crash — the forward clamps counts to capacity); deployment on a
    very different input distribution should re-measure and re-tune.
    """
    import os as _os
    _override = _os.environ.get("GRASPGENX_CAPS")
    if _override:
        # Comma-separated per-stage capacities for experiments (e.g. tuning the
        # padding down for real scene objects, which pool far below the default).
        sched = [int(x) for x in _override.split(",")][:num_stages]
        assert sched[0] == num_points, "stage-0 capacity must equal num_points"
        return sched

    # The input point cloud is always num_points (=3500 in the release config),
    # so the pooling is bounded by how that fixed budget distributes into voxels.
    # Measured across the real demo objects (3 resamples each) the pooling stages
    # keep at most [1380, 610, 212, 63] real points; these caps sit ~1.5-2x above
    # that, so no real point is dropped while the deep-stage padded compute (the
    # runtime dominator) shrinks 3-4x. Validated within the encoder's own shuffle
    # noise floor (24.5 vs 67 mm; 8.8/10 top-10; 97.6% threshold agreement) and
    # ~1.4x faster on the everything-engine. The previous, looser schedule
    # ([3500,3500,3072,1536,512]) carried large unused headroom.
    base_n = 3500
    base = [3500, 2048, 1024, 384, 128]
    if num_points == base_n:
        sched = base[:num_stages]
    else:
        r = num_points / base_n
        sched = [min(num_points, max(1, round(c * r))) for c in base[:num_stages]]
        sched[0] = num_points
    return sched


def _row_where(valid: torch.Tensor, a: torch.Tensor, fill: float = 0.0):
    """Zero (or fill) the rows of `a` where valid is False. valid: [N] bool."""
    return torch.where(valid.unsqueeze(-1), a, torch.full_like(a, fill))


class PTv3VanillaExport(nn.Module):
    """Static-shape forward over an existing PointTransformerV3Vanilla's
    weights. One object per call.

    forward(feat [N,3], coord [N,3], grid_coord [N,3] int32,
            code [O,N] int64, order [O,N] int64, inverse [O,N] int64)
        -> [1, output_dim]
    """

    def __init__(
        self,
        encoder: PointTransformerV3Vanilla,
        num_points: int,
        plugin_ops: bool = False,
        fp32_attn: bool = None,
        capacities: list = None,
    ):
        super().__init__()
        self.encoder = encoder  # shared weights, no copy
        self.N = num_points
        self.num_orders = len(encoder.order)
        # Attention precision on the plugin/TRT path. Default (False): use the
        # graspgenx::flash_attn QDP plugin (torch fp16 flash SDPA) — correct
        # (matches eager) and fast. Set True to force the manual fp32 path
        # instead (slower; kept as a fallback / for the non-plugin TRT build,
        # where TRT's own fp16 SDPA diverges).
        self.fp32_attn = bool(fp32_attn) if fp32_attn is not None else False
        # plugin_ops routes scatter-reduce / 1-D argsort through the
        # graspgenx torch custom ops backed by TensorRT QDP plugins
        # (graspgenx/models/trt_plugins.py) — numerically identical (the
        # eager impls ARE the plain torch ops); required for a pure
        # single-engine TRT build. Default False keeps native ops
        # (torch.export / ONNX path).
        self.plugin_ops = plugin_ops
        if plugin_ops:
            import graspgenx.models.trt_plugins  # noqa: F401  registers the ops

        # Static stage structure: [(pooling|None, [blocks...]), ...]
        self._stages = []
        for _name, stage in encoder.enc._modules.items():
            pool = None
            blocks = []
            for _bname, mod in stage._modules.items():
                if isinstance(mod, VanillaSerializedPooling):
                    pool = mod
                elif isinstance(mod, VanillaBlock):
                    blocks.append(mod)
            self._stages.append((pool, blocks))

        # Per-stage working capacity. Pooling stages shrink the point count,
        # so the deep stages carry far fewer real points than N; slicing each
        # stage's buffers to a fixed smaller capacity cuts the padded compute
        # in the expensive deep blocks (which dominate runtime). A stage keeps
        # only the first cap[s] slots after pooling — safe as long as the
        # pooled valid count n_new <= cap[s].
        #
        # Default (capacities=None): NO reduction (full N at every stage) —
        # keeps the ONNX / torch.export path and parity tests exact. The TRT
        # engine build opts in via ``reduced_capacity_schedule`` (see
        # build_everything_graph / specs/trt_plugins.md), which nearly halves
        # encoder latency at ~3e-3 embedding cost (well under the encoder's
        # own ~2.7e-2 shuffle drift).
        S = len(self._stages)
        self.capacities = [num_points] * S if capacities is None else list(capacities)
        assert len(self.capacities) == S
        assert self.capacities[0] == num_points  # enc0 has no pooling

    def _argsort1d(self, x):
        if self.plugin_ops:
            return torch.ops.graspgenx.argsort(x)
        return torch.argsort(x)

    def _gather0(self, src, index):
        """src[index] along dim 0. torch_tensorrt 2.6 mis-converts
        aten.index.Tensor for a float src (feeds the float tensor as TRT's
        gather index), so the plugin path routes through a QDP gather."""
        if self.plugin_ops:
            return torch.ops.graspgenx.gather0(src, index)
        return src[index]

    def _scatter_reduce(self, dst, index, src, reduce):
        """Functional scatter-reduce over dim 0, include_self=True (the init
        values make this equal to include_self=False, see _pool)."""
        if self.plugin_ops:
            return torch.ops.graspgenx.scatter_reduce(dst, index, src, reduce)
        return torch.scatter_reduce(dst, 0, index, src, reduce=reduce, include_self=True)

    # ── hash sparse conv with an ONNX-safe searchsorted ─────────────────────

    def _hash_conv(self, conv, feat, grid_coord, batch):
        """HashSparseConv3d.forward with `torch.searchsorted` replaced by an
        exact cat-sort + cumsum lookup (aten.searchsorted has no ONNX
        mapping). Table keys are doubled and query keys doubled+1, so equal
        keys interleave strictly (table first) and sort stability is
        irrelevant. `cumsum(is_table)-1` then yields, at every sorted
        position, the index of the latest table entry — i.e. the matching
        slot for an equal query. (Under an in-table hash collision this picks
        the last colliding voxel where eager picks the first; hash collisions
        between distinct voxels are equally undefined in eager.)"""
        if self.plugin_ops:
            # Whole neighbor-lookup sub-block as one QDP plugin — keeps every
            # int64 glue op (sort / cumsum / eq mask / index gathers) out of
            # the TRT graph, where torch_tensorrt 2.6 mis-converts them.
            bias = (
                conv.bias
                if conv.bias is not None
                else feat.new_zeros(conv.weight.shape[2])
            )
            return torch.ops.graspgenx.hash_conv(
                feat, grid_coord.int(), batch, conv.weight, bias, conv.offsets
            )

        N = feat.shape[0]
        K3 = conv.offsets.shape[0]
        device = feat.device

        keys = conv._hash(batch, grid_coord)
        sort_idx = self._argsort1d(keys)
        sorted_keys = keys[sort_idx]

        neighbor_coords = grid_coord.unsqueeze(1) + conv.offsets.unsqueeze(0)
        neighbor_batch = batch.unsqueeze(1).expand(-1, K3)
        neighbor_keys = conv._hash(
            neighbor_batch.reshape(-1), neighbor_coords.reshape(-1, 3)
        )  # (N * K3,)
        Q = neighbor_keys.shape[0]

        merged = torch.cat([sorted_keys * 2, neighbor_keys * 2 + 1])
        morder = self._argsort1d(merged)
        is_table = (morder < N).long()
        cnt = torch.cumsum(is_table, dim=0) - 1  # latest table slot per position
        inv_perm = torch.zeros_like(morder)
        inv_perm.scatter_(0, morder, torch.arange(N + Q, device=device))
        q_pos = inv_perm[N:]
        positions = cnt[q_pos].clamp(min=0, max=N - 1)
        found_mask = (cnt[q_pos] >= 0) & (sorted_keys[positions] == neighbor_keys)
        found_indices = sort_idx[positions]

        neighbor_feat = self._gather0(feat, found_indices) * found_mask.unsqueeze(-1)
        neighbor_feat = neighbor_feat.view(N, K3, conv.in_channels)
        out = torch.einsum("nki,kio->no", neighbor_feat, conv.weight)
        if conv.bias is not None:
            out = out + conv.bias
        return out

    # ── serialized attention (regime-exact) ─────────────────────────────────

    def _attention(self, attn, feat, order_o, inverse_o, n, valid):
        N = feat.shape[0]  # current stage capacity (may be < self.N)
        K = attn.patch_size_max
        H = attn.num_heads
        C = attn.channels
        P = (N + K - 1) // K  # static patch count
        device = feat.device

        idx = torch.arange(P * K, device=device)
        regime_dup = n > K  # 0-dim bool tensor
        n_pad = torch.div(n + K - 1, K, rounding_mode="trunc") * K
        # Layout: position i holds serialized position pad[i].
        # i < n: identity. n <= i < n_pad (only when n > K): duplicate the
        # previous patch's tail (i - K), exactly as eager. Else: garbage -> 0.
        pad = torch.where(
            idx < n,
            idx,
            torch.where(
                regime_dup & (idx < n_pad),
                (idx - K).clamp(min=0),
                torch.zeros_like(idx),
            ),
        )

        gather = order_o.clamp(min=0, max=N - 1)[pad]  # [P*K] point indices
        qkv = self._gather0(attn.qkv(feat), gather)  # [P*K, 3C]
        q, k, v = (
            qkv.reshape(P, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )

        # Additive key mask: in the n <= K regime eager attends over exactly
        # n unpadded tokens -> mask key slots >= n. In the n > K regime eager
        # uses duplication (keys all real) -> no mask. Finite -3e4 underflows
        # to exact zero after softmax, and keeps fully-garbage patches NaN-free.
        key_mask = (~regime_dup) & (idx >= n)
        bias = torch.where(
            key_mask, torch.tensor(_NEG_MASK, device=device), torch.tensor(0.0, device=device)
        ).reshape(P, 1, 1, K)

        if self.plugin_ops and not self.fp32_attn:
            # fp16 flash SDPA via QDP plugin (torch's kernel) — matches the
            # eager encoder exactly, unlike TRT's own fp16 SDPA which diverges.
            # Correct AND fast; avoids the fp32-attention cost.
            out = torch.ops.graspgenx.flash_attn(q, k, v, bias)
            out = out.transpose(1, 2).reshape(-1, C)
        elif attn.enable_flash and not self.fp32_attn:
            input_dtype = q.dtype
            q, k, v = q.half(), k.half(), v.half()
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.half(), scale=attn.scale
            )
            out = out.to(input_dtype).transpose(1, 2).reshape(-1, C)
        else:
            # Manual fp32 attention (backend-deterministic). Used for the
            # pure-TRT path; mathematically the same softmax attention.
            q, k = q.float(), k.float()
            logits = (q * attn.scale) @ k.transpose(-2, -1)
            logits = logits + bias
            logits = logits.float()
            w = attn.softmax(logits).to(v.dtype)
            out = (w @ v).transpose(1, 2).reshape(-1, C)

        inv = inverse_o.clamp(min=0, max=P * K - 1)
        feat_out = self._gather0(out, inv)  # [N, C]
        feat_out = attn.proj(feat_out)
        return _row_where(valid, feat_out)

    def _block(self, block, feat, grid_coord, order_o, inverse_o, n, valid, batch):
        # CPE (fixed-shape hash conv; invalid grid coords are far-away
        # sentinels so they never match a real neighbor).
        cpe = self._hash_conv(block.cpe_conv, feat, grid_coord, batch)
        cpe = block.cpe_norm(block.cpe_linear(cpe))
        feat = feat + _row_where(valid, cpe)

        shortcut = feat
        x = block.norm1(feat) if block.pre_norm else feat
        x = self._attention(block.attn, x, order_o, inverse_o, n, valid)
        feat = shortcut + x  # eval: drop_path = identity

        shortcut = feat
        x = block.norm2(feat) if block.pre_norm else feat
        x = block.mlp(x)
        feat = shortcut + _row_where(valid, x)
        return _row_where(valid, feat)

    # ── fixed-capacity pooling (unique -> sort/flags/cumsum/scatter) ────────

    def _pool(self, pool, feat, coord, grid_coord, code, n, valid):
        N = feat.shape[0]  # input stage capacity
        device = feat.device
        arange = torch.arange(N, device=device)
        sentinel = _SENTINEL_BASE + arange  # distinct, sorts to the end

        # pooling_depth = 1 (stride 2); floor-div == >>3 for non-negative codes
        code = torch.div(code, 8, rounding_mode="floor")
        c0 = torch.where(valid, code[0], sentinel)
        valid_sorted = arange < n  # invalid slots sort to the end

        if self.plugin_ops:
            # All integer clustering glue as one single-output plugin (keeps
            # the ne / & / cumsum / int gathers out of the TRT graph). Returns
            # one packed int64 tensor; unpack with static slices + reshape
            # (both TRT-convertible) — single-output avoids torch_tensorrt's
            # multi-output-plugin bug.
            O = self.num_orders
            packed = torch.ops.graspgenx.pool_cluster(c0, code, n)
            sidx = packed[0:N]
            cluster = packed[N : 2 * N]
            head_idx = packed[2 * N : 3 * N]
            n_new = packed[3 * N]
            base = 3 * N + 1
            new_code = packed[base : base + O * N].reshape(O, N)
            new_order = packed[base + O * N : base + 2 * O * N].reshape(O, N)
            new_inverse = packed[base + 2 * O * N : base + 3 * O * N].reshape(O, N)
        else:
            sidx = self._argsort1d(c0)
            sorted_c = c0[sidx]
            prev = torch.cat([sorted_c.new_full((1,), -1), sorted_c[:-1]])
            flags = (sorted_c != prev) & valid_sorted
            cluster = torch.cumsum(flags.long(), dim=0) - 1
            cluster = cluster.clamp(min=0)
            # Invalid rows -> trash slot N-1 (only trash when free).
            cluster = torch.where(
                valid_sorted, cluster, torch.full_like(cluster, N - 1)
            )
            n_new = flags.sum()
        valid_new = arange < n_new

        cluster_e = cluster.unsqueeze(-1)

        # Feature reduce — replicates segment_csr_vanilla(reduce=pool.reduce).
        proj = pool.proj(feat)
        src = self._gather0(proj, sidx)
        # (include_self=True == False given these init values.)
        if pool.reduce == "max":
            pooled = self._scatter_reduce(
                torch.full(
                    (N, proj.shape[1]), float("-inf"), device=device, dtype=proj.dtype
                ),
                cluster_e.expand_as(src),
                torch.where(
                    valid_sorted.unsqueeze(-1), src,
                    torch.full_like(src, float("-inf")),
                ),
                "amax",
            )
        elif pool.reduce in ("mean", "sum"):
            pooled = self._scatter_reduce(
                torch.zeros(N, proj.shape[1], device=device, dtype=proj.dtype),
                cluster_e.expand_as(src),
                _row_where(valid_sorted, src),
                pool.reduce,
            )
        else:  # pragma: no cover - release uses max
            raise NotImplementedError(pool.reduce)
        pooled = _row_where(valid_new, pooled)

        # Coord mean.
        csrc = _row_where(valid_sorted, self._gather0(coord, sidx))
        csum = self._scatter_reduce(
            torch.zeros(N, 3, device=device, dtype=coord.dtype),
            cluster_e.expand_as(csrc), csrc, "sum",
        )
        counts = self._scatter_reduce(
            torch.zeros(N, device=device, dtype=coord.dtype),
            cluster, valid_sorted.to(coord.dtype), "sum",
        )
        pooled_coord = csum / counts.clamp(min=1.0).unsqueeze(-1)
        pooled_coord = _row_where(valid_new, pooled_coord)

        if not self.plugin_ops:
            # Head index per cluster (first occurrence in sorted order).
            head_pos = self._scatter_reduce(
                torch.full((N,), N - 1, device=device, dtype=torch.long),
                cluster, arange, "amin",
            )
            head_idx = sidx[head_pos.clamp(max=N - 1)]
            new_code = code[:, head_idx]
            new_code = torch.where(
                valid_new.unsqueeze(0), new_code, sentinel.unsqueeze(0)
            )
            new_order = torch.stack(
                [self._argsort1d(new_code[o]) for o in range(self.num_orders)]
            )
            new_inverse = torch.zeros_like(new_order).scatter_(
                dim=1,
                index=new_order,
                src=arange.unsqueeze(0).expand(self.num_orders, -1),
            )
        # else: head_idx / new_code / new_order / new_inverse came from the
        # pool_cluster plugin above.

        # New grid coords; invalid rows -> far sentinels. (floor-div ==
        # right-shift for non-negative coords; TRT has no bitwise-shift.)
        new_grid = torch.div(
            self._gather0(grid_coord, head_idx), 2, rounding_mode="floor"
        ).to(grid_coord.dtype)
        far = _FAR_GRID + arange * 4  # distinct, far from real coords, int32-safe
        new_grid = torch.where(
            valid_new.unsqueeze(-1), new_grid, far.unsqueeze(-1).to(new_grid.dtype)
        )

        if pool.norm is not None:
            pooled = pool.norm(pooled)
        if pool.act is not None:
            pooled = pool.act(pooled)
        pooled = _row_where(valid_new, pooled)

        return pooled, pooled_coord, new_grid, new_code, new_order, new_inverse, n_new, valid_new

    # ── batched forward (M objects) ─────────────────────────────────────────
    #
    # Same math as the single-object path, vectorized over a leading batch dim
    # M so the two PtV3 encoders process a whole scene's objects in one call.
    # The batching win is real: the torch encoder's per-object cost drops ~2.2x
    # from M=1 to M=8 (fewer, larger GEMMs; amortized launch overhead). Objects
    # stay independent: hash_conv separates them by the per-object `batch` index
    # (reused via a flat [M*N] call), attention patches and pooling clusters are
    # kept per-object by the leading M dim, and the final mean-pool is per row.

    @staticmethod
    def _row_where_b(valid, a, fill: float = 0.0):
        """Zero (or fill) rows where valid is False. valid [M,N], a [M,N,C]."""
        return torch.where(valid.unsqueeze(-1), a, torch.full_like(a, fill))

    @staticmethod
    def _norm_b(norm, x):
        """Apply a norm over a [M,N,C] batch. BatchNorm1d treats dim 1 as the
        channel, so flatten to [M*N,C] first (in eval it's a per-channel affine,
        identical to the per-object path); LayerNorm on the last dim is
        unchanged by the flatten."""
        if norm is None:
            return x
        M, N, C = x.shape
        return norm(x.reshape(M * N, C)).reshape(M, N, C)

    def _attention_b(self, attn, feat, order_o, inverse_o, n, valid):
        # feat [M,N,C]; order_o/inverse_o [M,N]; n [M]; valid [M,N]
        M, N, C = feat.shape
        K = attn.patch_size_max
        H = attn.num_heads
        P = (N + K - 1) // K
        device = feat.device

        idx = torch.arange(P * K, device=device)[None].expand(M, -1)  # [M, P*K]
        nb = n[:, None]
        regime_dup = (n > K)[:, None]  # [M,1]
        n_pad = torch.div(n + K - 1, K, rounding_mode="trunc")[:, None] * K
        pad = torch.where(
            idx < nb, idx,
            torch.where(regime_dup & (idx < n_pad), (idx - K).clamp(min=0),
                        torch.zeros_like(idx)),
        )  # [M, P*K] local indices

        order_c = order_o.clamp(min=0, max=N - 1)
        gather = torch.gather(order_c, 1, pad)  # [M, P*K] local indices
        qkv_all = attn.qkv(feat)  # [M,N,3C]
        qkv = qkv_all.gather(1, gather.unsqueeze(-1).expand(-1, -1, qkv_all.shape[-1]))
        qkv = qkv.reshape(M * P, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # each [M*P, H, K, C//H]

        key_mask = (~regime_dup) & (idx >= nb)  # [M, P*K]
        bias = torch.where(
            key_mask, torch.full_like(key_mask, _NEG_MASK, dtype=torch.float32),
            torch.zeros_like(key_mask, dtype=torch.float32),
        ).reshape(M * P, 1, 1, K)

        q, k = q.float(), k.float()
        logits = (q * attn.scale) @ k.transpose(-2, -1) + bias
        w = attn.softmax(logits.float()).to(v.dtype)
        out = (w @ v).transpose(1, 2).reshape(M, P * K, C)

        inv = inverse_o.clamp(min=0, max=P * K - 1)
        feat_out = out.gather(1, inv.unsqueeze(-1).expand(-1, -1, C))
        feat_out = attn.proj(feat_out)
        return self._row_where_b(valid, feat_out)

    def _hash_conv_b(self, conv, feat, grid_coord):
        """Batched hash conv via the single-object kernel on a flat [M*N] stream
        with per-object batch indices (so no object sees another's neighbors)."""
        M, N = feat.shape[0], feat.shape[1]
        batch = torch.arange(M, device=feat.device).repeat_interleave(N)
        out = self._hash_conv(conv, feat.reshape(M * N, -1),
                              grid_coord.reshape(M * N, 3), batch)
        return out.reshape(M, N, -1)

    def _block_b(self, block, feat, grid_coord, order_o, inverse_o, n, valid):
        cpe = self._hash_conv_b(block.cpe_conv, feat, grid_coord)
        cpe = self._norm_b(block.cpe_norm, block.cpe_linear(cpe))
        feat = feat + self._row_where_b(valid, cpe)
        shortcut = feat
        x = self._norm_b(block.norm1, feat) if block.pre_norm else feat
        x = self._attention_b(block.attn, x, order_o, inverse_o, n, valid)
        feat = shortcut + x
        shortcut = feat
        x = self._norm_b(block.norm2, feat) if block.pre_norm else feat
        x = block.mlp(x)
        feat = shortcut + self._row_where_b(valid, x)
        return self._row_where_b(valid, feat)

    def _pool_b(self, pool, feat, coord, grid_coord, code, n, valid):
        M, N, _ = feat.shape
        device = feat.device
        O = self.num_orders
        arange = torch.arange(N, device=device)
        sentinel = _SENTINEL_BASE + arange  # [N]

        code = torch.div(code, 8, rounding_mode="floor")  # [M,O,N]
        c0 = torch.where(valid, code[:, 0], sentinel[None])  # [M,N]
        valid_sorted = arange[None] < n[:, None]  # [M,N] (sorted order)

        sidx = torch.argsort(c0, dim=1)
        sorted_c = torch.gather(c0, 1, sidx)
        prev = torch.cat([sorted_c.new_full((M, 1), -1), sorted_c[:, :-1]], dim=1)
        flags = (sorted_c != prev) & valid_sorted
        cluster = (torch.cumsum(flags.long(), dim=1) - 1).clamp(min=0)
        cluster = torch.where(valid_sorted, cluster, torch.full_like(cluster, N - 1))
        n_new = flags.sum(dim=1)  # [M]
        valid_new = arange[None] < n_new[:, None]

        proj = pool.proj(feat)  # [M,N,Cout]
        Cout = proj.shape[-1]
        src = torch.gather(proj, 1, sidx[..., None].expand(-1, -1, Cout))
        cluster_e = cluster[..., None].expand(-1, -1, Cout)
        if pool.reduce == "max":
            neg = torch.where(valid_sorted[..., None], src, torch.full_like(src, float("-inf")))
            pooled = torch.full((M, N, Cout), float("-inf"), device=device, dtype=proj.dtype)
            pooled = pooled.scatter_reduce(1, cluster_e, neg, reduce="amax", include_self=True)
        elif pool.reduce in ("mean", "sum"):
            pooled = torch.zeros(M, N, Cout, device=device, dtype=proj.dtype)
            pooled = pooled.scatter_reduce(1, cluster_e, self._row_where_b(valid_sorted, src),
                                           reduce=pool.reduce, include_self=True)
        else:  # pragma: no cover
            raise NotImplementedError(pool.reduce)
        pooled = self._row_where_b(valid_new, pooled)

        csrc = self._row_where_b(
            valid_sorted, torch.gather(coord, 1, sidx[..., None].expand(-1, -1, 3))
        )
        csum = torch.zeros(M, N, 3, device=device, dtype=coord.dtype).scatter_reduce(
            1, cluster[..., None].expand(-1, -1, 3), csrc, reduce="sum", include_self=True)
        counts = torch.zeros(M, N, device=device, dtype=coord.dtype).scatter_reduce(
            1, cluster, valid_sorted.to(coord.dtype), reduce="sum", include_self=True)
        pooled_coord = self._row_where_b(valid_new, csum / counts.clamp(min=1.0)[..., None])

        head_pos = torch.full((M, N), N - 1, device=device, dtype=torch.long).scatter_reduce(
            1, cluster, arange[None].expand(M, N), reduce="amin", include_self=True)
        head_idx = torch.gather(sidx, 1, head_pos.clamp(max=N - 1))  # [M,N]
        new_code = torch.gather(code, 2, head_idx[:, None, :].expand(-1, O, -1))
        new_code = torch.where(valid_new[:, None], new_code, sentinel[None, None])
        new_order = torch.argsort(new_code, dim=2)
        new_inverse = torch.zeros_like(new_order).scatter_(
            2, new_order, arange[None, None].expand(M, O, N))

        new_grid = torch.div(
            torch.gather(grid_coord, 1, head_idx[..., None].expand(-1, -1, 3)),
            2, rounding_mode="floor").to(grid_coord.dtype)
        far = (_FAR_GRID + arange * 4)
        new_grid = torch.where(
            valid_new[..., None], new_grid, far[None, :, None].to(new_grid.dtype)
        )

        if pool.norm is not None:
            pooled = self._norm_b(pool.norm, pooled)
        if pool.act is not None:
            pooled = pool.act(pooled)
        pooled = self._row_where_b(valid_new, pooled)
        return pooled, pooled_coord, new_grid, new_code, new_order, new_inverse, n_new, valid_new

    def forward_batched(self, feat, coord, grid_coord, code, order, inverse):
        """Batched encoder. feat/coord/grid_coord [M,N,3]; code/order/inverse
        [M,O,N] (int). Returns [M, output_dim]."""
        M, N = feat.shape[0], self.N
        device = feat.device
        n = torch.full((M,), N, dtype=torch.long, device=device)
        valid = torch.ones(M, N, dtype=torch.bool, device=device)

        emb: VanillaEmbedding = self.encoder.embedding
        feat = self._hash_conv_b(emb.conv, feat, grid_coord)
        feat = self._norm_b(emb.norm, feat)
        if emb.act is not None:
            feat = emb.act(feat)

        for s, (pool, blocks) in enumerate(self._stages):
            if pool is not None:
                (feat, coord, grid_coord, code, order, inverse, n, valid) = self._pool_b(
                    pool, feat, coord, grid_coord, code, n, valid)
                cap = self.capacities[s]
                if cap < feat.shape[1]:
                    feat = feat[:, :cap]; coord = coord[:, :cap]; grid_coord = grid_coord[:, :cap]
                    code = code[:, :, :cap]
                    order = order[:, :, :cap].clamp(max=cap - 1)
                    inverse = inverse[:, :, :cap].clamp(max=cap - 1)
                    n = torch.clamp(n, max=cap)
                    valid = torch.arange(cap, device=device)[None] < n[:, None]
            for block in blocks:
                o = block.attn.order_index
                feat = self._block_b(block, feat, grid_coord, order[:, o], inverse[:, o], n, valid)

        pooled = self._row_where_b(valid, feat).sum(dim=1) / n.to(feat.dtype)[:, None]  # [M,C]
        return self.encoder.projection(pooled)

    # ── forward (single object) ────────────────────────────────────────────

    def forward(self, feat, coord, grid_coord, code, order, inverse):
        N = self.N
        device = feat.device
        batch = torch.zeros(N, dtype=torch.long, device=device)
        n = torch.full((), N, dtype=torch.long, device=device)
        valid = torch.ones(N, dtype=torch.bool, device=device)

        emb: VanillaEmbedding = self.encoder.embedding
        feat = self._hash_conv(emb.conv, feat, grid_coord, batch)
        if emb.norm is not None:
            feat = emb.norm(feat)
        if emb.act is not None:
            feat = emb.act(feat)

        for s, (pool, blocks) in enumerate(self._stages):
            if pool is not None:
                (feat, coord, grid_coord, code, order, inverse, n, valid) = self._pool(
                    pool, feat, coord, grid_coord, code, n, valid
                )
                cap = self.capacities[s]
                if cap < feat.shape[0]:
                    # Compact to this stage's capacity. Pooled valid points
                    # occupy slots [0:n_new] with n_new <= cap, so slicing
                    # keeps every real point; order/inverse are clamped into
                    # range (out-of-range entries index masked-out slots).
                    feat = feat[:cap]
                    coord = coord[:cap]
                    grid_coord = grid_coord[:cap]
                    code = code[:, :cap]
                    order = order[:, :cap].clamp(max=cap - 1)
                    inverse = inverse[:, :cap].clamp(max=cap - 1)
                    batch = batch[:cap]
                    # Cap the valid count at the capacity: if pooling kept more
                    # than cap points (a pathologically spread cloud), the
                    # overflow slots were just sliced away, so clamp n to stay
                    # consistent and keep all downstream indices in range. With
                    # adequate headroom n <= cap and this is a no-op.
                    n = torch.clamp(n, max=cap)
                    valid = torch.arange(cap, device=device) < n
            for block in blocks:
                o = block.attn.order_index
                feat = self._block(
                    block, feat, grid_coord, order[o], inverse[o], n, valid, batch
                )

        pooled = _row_where(valid, feat).sum(dim=0, keepdim=True) / n.to(feat.dtype)
        return self.encoder.projection(pooled)
