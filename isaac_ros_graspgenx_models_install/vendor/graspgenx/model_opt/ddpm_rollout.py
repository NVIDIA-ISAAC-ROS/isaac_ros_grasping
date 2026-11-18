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

"""Loop-free, diffusers-free DDPM rollout for single-graph export.

The eager reverse-diffusion path in :class:`GraspGenGenerator` runs a Python
``for`` loop over ``num_diffusion_iters_eval`` timesteps, calling
``diffusers.DDPMScheduler.step`` on the host each iteration. That loop cannot
be captured by ``torch.export`` / ONNX / TensorRT, so the sampler cannot ship
as one engine.

This module reimplements the exact same math with two properties:

1. **No diffusers at runtime.** All schedule coefficients are precomputed at
   construction time by replicating ``DDPMScheduler`` (the pinned diffusers
   version) operation-for-operation in float32, so results are bit-identical
   to the eager loop given the same noise. See ``tests/test_ddpm_rollout_parity.py``.
2. **No Python loop in the exported graph.** The per-step update is a single
   fused equation with per-step constant coefficients; ``forward`` iterates a
   fixed Python-int trip count, which tracing/`torch.export` *unrolls* into
   one static graph. (The DDPM recursion is inherently sequential — each step
   feeds the denoiser the previous step's output and ``clip_sample`` is a
   per-step nonlinearity — so unrolling, not a closed form, is the exact
   loop-free formulation.)

Per-step update, replicating ``DDPMScheduler.step`` with
``prediction_type='epsilon'`` and ``variance_type='fixed_small'``::

    x0      = clamp((x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t), -1, 1)
    c_x0    = sqrt(abar_{t-1}) * beta_t / (1 - abar_t)
    c_xt    = sqrt(alpha_t) * (1 - abar_{t-1}) / (1 - abar_t)
    sigma_t = sqrt(max((1 - abar_{t-1}) / (1 - abar_t) * beta_t, 1e-20))
    x_{t-1} = c_x0 * x0 + c_xt * x_t + sigma_t * z_t        (sigma_0 = 0)

Noise (``init_noise``, ``step_noise``) is an explicit *input* — never drawn
inside the module — which keeps the exported graph deterministic and makes
numerical-parity testing against the eager loop possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def make_ddpm_betas(
    beta_schedule: str,
    num_train_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Beta schedule, replicated from ``diffusers.DDPMScheduler.__init__``."""
    if beta_schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    if beta_schedule == "scaled_linear":
        return (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        )
    if beta_schedule == "squaredcos_cap_v2":
        # Glide cosine schedule (diffusers' betas_for_alpha_bar, max_beta=0.999).
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_train_timesteps):
            t1 = i / num_train_timesteps
            t2 = (i + 1) / num_train_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)
    raise NotImplementedError(f"beta_schedule {beta_schedule!r} is not supported")


@dataclass
class DDPMStepCoeffs:
    """Per-eval-step scalar coefficients for one DDPM schedule.

    All tensors have shape ``[num_inference_steps]``, float32, indexed by eval
    position (position 0 = largest timestep). Computed with the exact float32
    operation sequence of ``DDPMScheduler.step`` so the eager loop and the
    unrolled rollout agree bitwise.
    """

    timesteps: List[int]  # descending, e.g. [19, 18, ..., 0]
    c_eps: torch.Tensor  # sqrt(1 - abar_t)
    c_sqrt_alpha: torch.Tensor  # sqrt(abar_t)   (divisor for pred-x0)
    c_x0: torch.Tensor  # coefficient on clipped pred-x0
    c_xt: torch.Tensor  # coefficient on the current sample
    sigma: torch.Tensor  # stddev of the added noise (0 at t == 0)

    @classmethod
    def from_schedule(
        cls,
        beta_schedule: str,
        num_train_timesteps: int,
        num_inference_steps: int,
    ) -> "DDPMStepCoeffs":
        betas = make_ddpm_betas(beta_schedule, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        one = torch.tensor(1.0)

        # Timestep spacing exactly as DDPMScheduler.set_timesteps.
        num_inference_steps = min(num_train_timesteps, num_inference_steps)
        step_ratio = num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_train_timesteps, step_ratio)[::-1].copy()

        c_eps, c_sqrt_alpha, c_x0, c_xt, sigma = [], [], [], [], []
        for t in timesteps.tolist():
            alpha_prod_t = alphas_cumprod[t]
            # NB: DDPMScheduler.step uses t-1 (not the previous *eval*
            # timestep) even when eval steps are strided; replicated as-is.
            alpha_prod_t_prev = alphas_cumprod[t - 1] if t > 0 else one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            c_eps.append(beta_prod_t ** (0.5))
            c_sqrt_alpha.append(alpha_prod_t ** (0.5))
            c_x0.append((alpha_prod_t_prev ** (0.5) * betas[t]) / beta_prod_t)
            c_xt.append(alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t)
            if t > 0:
                variance = torch.clamp(
                    (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * betas[t], min=1e-20
                )
                sigma.append(variance ** (0.5))
            else:
                sigma.append(torch.tensor(0.0))

        return cls(
            timesteps=timesteps.tolist(),
            c_eps=torch.stack(c_eps),
            c_sqrt_alpha=torch.stack(c_sqrt_alpha),
            c_x0=torch.stack(c_x0),
            c_xt=torch.stack(c_xt),
            sigma=torch.stack(sigma),
        )


class UnrolledDDPMRollout(nn.Module):
    """Full reverse-diffusion rollout as one traceable module.

    Contains the denoiser weights (``diffusion_head``) so the exported graph
    is self-contained: N denoiser evaluations + N scheduler updates, unrolled.

    Supports both generator configurations:
    - compositional: position dims [0:3) use one schedule, rotation dims
      [3:output_dim) another (coefficients are concatenated per-dim, so the
      fused update handles both slices in one elementwise op);
    - single-scheduler: one coefficient set across all dims.

    Inputs:
        obs_embedding: ``[B, obs_dim]`` object+gripper conditioning.
        init_noise:    ``[B, output_dim]`` x_T ~ N(0, I).
        step_noise:    ``[N, B, output_dim]`` per-step z_t ~ N(0, I); the last
                       row (t == 0) is unused since sigma_0 = 0.
    Output:
        ``[B, output_dim]`` denoised sample (same representation the eager
        loop yields before ``rt_to_matrix``).
    """

    def __init__(
        self,
        diffusion_head: nn.Module,
        *,
        output_dim: int,
        num_train_timesteps: int,
        num_inference_steps: int,
        compositional: bool = False,
        beta_schedule: Optional[str] = None,
        pos_beta_schedule: str = "scaled_linear",
        rot_beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        pos_dims: int = 3,
    ):
        super().__init__()
        self.diffusion_head = diffusion_head
        self.output_dim = output_dim
        self.clip_sample = clip_sample

        if compositional:
            pos = DDPMStepCoeffs.from_schedule(
                pos_beta_schedule, num_train_timesteps, num_inference_steps
            )
            rot = DDPMStepCoeffs.from_schedule(
                rot_beta_schedule, num_train_timesteps, num_inference_steps
            )
            assert pos.timesteps == rot.timesteps
            self._timesteps = pos.timesteps

            def per_dim(name):
                a = getattr(pos, name)[:, None].expand(-1, pos_dims)
                b = getattr(rot, name)[:, None].expand(-1, output_dim - pos_dims)
                return torch.cat([a, b], dim=1).contiguous()

        else:
            if beta_schedule is None:
                raise ValueError("beta_schedule is required when compositional=False")
            single = DDPMStepCoeffs.from_schedule(
                beta_schedule, num_train_timesteps, num_inference_steps
            )
            self._timesteps = single.timesteps

            def per_dim(name):
                return (
                    getattr(single, name)[:, None].expand(-1, output_dim).contiguous()
                )

        # [N, output_dim] per-step, per-dim constants. persistent=False keeps
        # them out of state_dict (they are derived, not learned).
        for name in ("c_eps", "c_sqrt_alpha", "c_x0", "c_xt", "sigma"):
            self.register_buffer(name, per_dim(name), persistent=False)

    @property
    def num_steps(self) -> int:
        return len(self._timesteps)

    def forward(
        self,
        obs_embedding: torch.Tensor,
        init_noise: torch.Tensor,
        step_noise: torch.Tensor,
    ) -> torch.Tensor:
        x = init_noise
        # Fixed-trip-count loop over Python ints: unrolled at trace/export
        # time into one static graph — no Python at runtime. The timestep
        # tensor is derived from the input (full_like) so the batch dim stays
        # dynamic under trace-based exporters.
        for i in range(self.num_steps):
            ts = torch.full_like(obs_embedding[:, 0], float(self._timesteps[i]))
            eps = self.diffusion_head(obs_embedding, ts, x)
            x0 = (x - self.c_eps[i] * eps) / self.c_sqrt_alpha[i]
            if self.clip_sample:
                x0 = torch.clamp(x0, -1, 1)
            x = self.c_x0[i] * x0 + self.c_xt[i] * x
            if self._timesteps[i] > 0:
                x = x + self.sigma[i] * step_noise[i]
        return x


# ── Export-safe pose conversions ─────────────────────────────────────────────
# Value-identical to graspgenx.utils.transformations.rt_to_matrix/matrix_to_rt
# (verified in tests/test_ddpm_rollout_parity.py), but written functionally:
# the originals use in-place slice writes (rt_to_matrix, so3.hat) and a
# boolean-mask gather (matrix_to_quaternion) that the TensorRT converter
# rejects for dynamic batch sizes.


def _sqrt_positive_part_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0, torch.sqrt(torch.clamp(x, min=0.0)), torch.zeros_like(x)
    )


def _sinc_fn(x: torch.Tensor) -> torch.Tensor:
    """torch.sinc by definition (sin(pi x)/(pi x), 1 at x=0) — aten::sinc has
    no ONNX opset mapping."""
    pix = torch.pi * x
    return torch.where(x == 0, torch.ones_like(x), torch.sin(pix) / pix)


def _axis_angle_to_quaternion_fn(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * _sinc_fn(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def _quaternion_to_axis_angle_fn(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * _sinc_fn(half_angles / torch.pi)
    return quaternions[..., 1:] / sin_half_angles_over_angles


def _matrix_to_quaternion_fn(matrix: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part_fn(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # One-hot-weighted sum instead of boolean-mask indexing: all candidates
    # are finite, so the zero-weighted terms vanish exactly.
    sel = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(matrix.dtype)[..., None]
    out = (quat_candidates * sel).sum(dim=-2)
    return torch.where(out[..., 0:1] < 0, -out, out)


def _rt_to_matrix_fn(
    sample: torch.Tensor, grasp_repr: str, kappa: Optional[float]
) -> torch.Tensor:
    from graspgenx.utils.transformations import (
        quaternion_to_matrix,
        rotation_6d_to_matrix,
    )

    t = sample[:, :3]
    if kappa is not None:
        t = t * (1.0 / kappa)
    if grasp_repr == "r3_so3":
        # so3.so3_exp_map returns axis_angle_to_matrix(x) = the quaternion
        # round-trip below; the skew-matrix side products it also computes are
        # unused for R.
        rot = quaternion_to_matrix(
            _axis_angle_to_quaternion_fn(sample[:, 3:] * torch.pi)
        )
    elif grasp_repr == "r3_6d":
        rot = rotation_6d_to_matrix(sample[:, 3:])
    else:
        raise NotImplementedError(f"grasp_repr {grasp_repr!r} not supported")
    top = torch.cat([rot, t.unsqueeze(-1)], dim=-1)  # [B, 3, 4]
    bottom = torch.cat(
        [torch.zeros_like(t), torch.ones_like(t[:, :1])], dim=-1
    ).unsqueeze(1)  # [B, 1, 4] = (0, 0, 0, 1)
    return torch.cat([top, bottom], dim=1)


def _matrix_to_rt_fn(
    mat: torch.Tensor, grasp_repr: str, kappa: Optional[float]
) -> torch.Tensor:
    from graspgenx.utils.transformations import matrix_to_rotation_6d

    t = mat[:, :3, 3]
    if kappa is not None:
        t = kappa * t
    rot_mat = mat[:, :3, :3]
    if grasp_repr == "r3_so3":
        rot = _quaternion_to_axis_angle_fn(_matrix_to_quaternion_fn(rot_mat))
        rot = rot * (1 / torch.pi)
    elif grasp_repr == "r3_6d":
        rot = matrix_to_rotation_6d(rot_mat)
    else:
        raise NotImplementedError(f"grasp_repr {grasp_repr!r} not supported")
    return torch.cat([t, rot], dim=-1).float()


class UnrolledRolloutWithScoring(nn.Module):
    """DDPM rollout + discriminator scoring as ONE traceable graph.

    Everything downstream of the two (unexportable) point-cloud encoders:

        gen_obs_embedding ─┐
        init_noise ────────┼─> unrolled DDPM rollout ─> sample [B, 6]
        step_noise ────────┘          |
                                      v
                     rt_to_matrix(gen_grasp_repr, gen_kappa) ─> grasps [B, 4, 4]
                                      |
                     matrix_to_rt(dis_grasp_repr, dis_kappa)
                                      |
        dis_obs_embedding ──> [sample_encoder | obs] ─> prediction_head
                                      |
                                      v
                          confidence = sigmoid(logits) [B, 1]

    The pose conversions use the export-safe functional equivalents above,
    value-identical to rt_to_matrix / matrix_to_rt, so results match running
    the generator fast path followed by GraspGenDiscriminator.forward.

    Inputs (B = num_objects * num_grasps, dynamic):
        gen_obs_embedding: [B, gen_obj_dim + gen_gripper_dim] generator
            conditioning (object embedding ‖ gripper embedding, already
            broadcast per grasp row).
        dis_obs_embedding: [B, dis_obj_dim + dis_gripper_dim] discriminator
            conditioning, same layout.
        init_noise:  [B, gen_output_dim]
        step_noise:  [N, B, gen_output_dim]
    Outputs:
        grasps:     [B, 4, 4] poses (translation in the centered object frame)
        confidence: [B, 1] discriminator sigmoid scores
    """

    def __init__(
        self,
        rollout: UnrolledDDPMRollout,
        sample_encoder: nn.Module,
        prediction_head: nn.Module,
        *,
        gen_grasp_repr: str,
        gen_kappa: Optional[float],
        dis_grasp_repr: str,
        dis_kappa: Optional[float],
    ):
        super().__init__()
        self.rollout = rollout
        self.sample_encoder = sample_encoder
        self.prediction_head = prediction_head
        self.gen_grasp_repr = gen_grasp_repr
        self.gen_kappa = gen_kappa
        self.dis_grasp_repr = dis_grasp_repr
        self.dis_kappa = dis_kappa

    @property
    def num_steps(self) -> int:
        return self.rollout.num_steps

    def forward(
        self,
        gen_obs_embedding: torch.Tensor,
        dis_obs_embedding: torch.Tensor,
        init_noise: torch.Tensor,
        step_noise: torch.Tensor,
    ):
        sample = self.rollout(gen_obs_embedding, init_noise, step_noise)
        grasps = _rt_to_matrix_fn(sample, self.gen_grasp_repr, self.gen_kappa)
        dis_input = _matrix_to_rt_fn(grasps, self.dis_grasp_repr, self.dis_kappa)
        sample_embedding = self.sample_encoder(dis_input)
        logits = self.prediction_head(
            torch.cat([sample_embedding, dis_obs_embedding], dim=-1)
        )
        return grasps, logits.sigmoid()


def build_fused_rollout_scoring(model) -> UnrolledRolloutWithScoring:
    """Construct the fused rollout+scoring module from a loaded ``GraspGen``
    model (generator + discriminator), unwrapping any TensorRT shims so the
    fused graph always embeds the eager weights.

    Only ``pose_repr == 'mlp'`` discriminators are supported (the release
    configuration): other pose representations transform gripper control
    points inside the scoring path.
    """
    discriminator = model.grasp_discriminator
    if discriminator.pose_repr != "mlp":
        raise ValueError(
            f"Fused scoring supports discriminator pose_repr='mlp' only "
            f"(got {discriminator.pose_repr!r})."
        )

    rollout = build_unrolled_rollout(model.grasp_generator)
    # Unwrap TensorRTModule shims installed by accelerate_sampler.
    sample_encoder = getattr(
        discriminator.sample_encoder, "eager_module", discriminator.sample_encoder
    )
    prediction_head = getattr(
        discriminator.prediction_head, "eager_module", discriminator.prediction_head
    )

    fused = UnrolledRolloutWithScoring(
        rollout,
        sample_encoder,
        prediction_head,
        gen_grasp_repr=model.grasp_generator.grasp_repr,
        gen_kappa=model.grasp_generator.kappa,
        dis_grasp_repr=discriminator.grasp_repr,
        dis_kappa=discriminator.kappa,
    )
    device = next(sample_encoder.parameters()).device
    return fused.to(device).eval()


def gripper_conditioning_input(module, data_batch) -> torch.Tensor:
    """The data-batch tensor feeding ``module.gripper_encoder``, per its
    gripper backbone (the sweep-volume-derivable backbones only)."""
    key = {
        "z_offset": "z_offset",
        "sweep_volume": "sweep_volume",
        "sweep_volume_v2": "sweep_volume_open_and_mid",
    }[module.gripper_backbone]
    return data_batch[key]


def compute_conditioning_embeddings(model, data_batch, num_objects, num_grasps):
    """Run both point-cloud encoders and build the fused graph's two
    conditioning inputs, broadcast per grasp row.

    This is exactly the PyTorch work that stays OUTSIDE the exported fused
    graph (the PTv3 encoders use ops with no ONNX/TRT equivalent).

    Returns (gen_obs_embedding [B, gen_dim], dis_obs_embedding [B, dis_dim])
    with B = num_objects * num_grasps; row i*num_grasps..(i+1)*num_grasps-1
    conditions the grasps of object i.
    """
    from graspgenx.models.model_utils import convert_to_ptv3_pc_format, offset2batch

    device = data_batch["points"].device
    offset = (
        torch.tensor([num_grasps]).repeat(num_objects).cumsum(dim=0).to(device)
    )
    mask = offset2batch(offset)
    n_pts = data_batch["points"].shape[-2]
    depth = data_batch["points"].reshape([-1, n_pts, 3])

    embeddings = []
    for module in (model.grasp_generator, model.grasp_discriminator):
        d = module.kappa * depth if module.kappa is not None else depth
        if module.object_backbone in ("ptv3", "ptv3vanilla"):
            d = convert_to_ptv3_pc_format(d, grid_size=module.grid_size)
        obj_emb = module.object_encoder(d)
        grip_emb = module.gripper_encoder(
            gripper_conditioning_input(module, data_batch)
        )
        embeddings.append(torch.cat([obj_emb[mask], grip_emb[mask]], dim=-1))
    return embeddings[0], embeddings[1]


def build_unrolled_rollout(generator) -> UnrolledDDPMRollout:
    """Construct an :class:`UnrolledDDPMRollout` mirroring a generator's
    scheduler configuration (compositional or single), using its *eager*
    denoiser (unwrapping any TensorRT shim).

    Only ``pose_repr == 'mlp'`` is supported: other pose representations
    re-encode the point cloud inside the loop and are out of scope.
    """
    if generator.pose_repr != "mlp":
        raise ValueError(
            f"Unrolled DDPM rollout supports pose_repr='mlp' only "
            f"(got {generator.pose_repr!r})."
        )

    head = generator.diffusion_head
    # Unwrap a TensorRTDiffusionHead installed by the per-step path.
    head = getattr(head, "eager_module", head)

    kwargs = dict(
        output_dim=generator.output_dim,
        num_train_timesteps=generator.num_diffusion_iters,
        num_inference_steps=generator.num_diffusion_iters_eval,
    )
    if generator.compositional_schedular:
        # Mirrors GraspGenGenerator.__init__: pos=scaled_linear,
        # rot=squaredcos_cap_v2, clip_sample=True (hardcoded there).
        rollout = UnrolledDDPMRollout(
            head, compositional=True, clip_sample=True, **kwargs
        )
    else:
        rollout = UnrolledDDPMRollout(
            head,
            compositional=False,
            beta_schedule=generator.beta_schedule,
            clip_sample=generator.clip_sample,
            **kwargs,
        )
    device = next(head.parameters()).device
    return rollout.to(device).eval()
