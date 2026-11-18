// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// CPU preprocessing for GraspGenX object point clouds:
//   1. Statistical outlier removal (SOR) using mean L1 k-neighbor distances.
//   2. Oriented bounding box (OBB) estimation using L2-SOR, a 2D convex hull,
//      a minimum-area rectangle, and 2nd/98th-percentile extents.
//   3. Grasp candidates sampled over the top and four side faces of the OBB
//      at the configured yaw angles, positions, and approach offsets.
// This stage generates candidates without scoring them.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace graspgenx {

struct GripperParams {
    float width_m = 0.085f;  // Maximum opening in meters; used by the OBB skip rule.
    float depth_m = 0.0f;    // Candidate base offset along the local approach axis.
};

struct MoeParams {
    int num_yaws = 36;
    std::vector<float> z_offsets_m = {-0.02f, 0.0f};
    float outlier_threshold = 0.014f;  // main SOR (L1) keep cutoff
    int outlier_k = 20;                // kNN K for both SORs
    float obb_position_spacing_m = 0.01f;
    bool obb_jitter = true;            // Gaussian point jitter with zero mean and 1e-4 m stddev.
};

struct Obb {
    std::array<float, 3> center{};
    std::array<float, 3> half_extent{};
    std::array<float, 9> R{};  // row-major 3x3 (world<-local)
    bool valid = false;
};

struct PreprocOut {
    std::vector<float> filtered_pc;  // [M,3] after the main SOR
    Obb obb;
    std::vector<float> candidates;   // [C,16] row-major 4x4 world-frame poses
    bool skipped_obb = false;
};

// Full preprocessing for one object. raw_pc: [N,3] row-major world-frame.
PreprocOut preprocess(float const* raw_pc, int N,
                      GripperParams const& g, MoeParams const& m);

// --- sub-steps exposed for validation + per-stage timing ---

// Main SOR (L1, absolute threshold). Returns the boolean keep-mask [N].
std::vector<uint8_t> sor_keep_mask_l1(float const* pc, int N, float threshold, int k);

// OBB from a world-frame cloud using L2-SOR and percentile extents.
Obb compute_obb(float const* pc, int N, MoeParams const& m);

// dense-topandside candidates for a given OBB. Returns [C,16] row-major 4x4.
// skipped set true when the auto skip-rule fires (all extents > gripper width).
std::vector<float> obb_candidates(Obb const& obb, GripperParams const& g,
                                  MoeParams const& m, bool& skipped);

}  // namespace graspgenx
