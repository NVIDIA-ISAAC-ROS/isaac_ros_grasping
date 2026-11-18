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

// CPU point-cloud serialization for the GraspGenX Point Transformer V3 encoder.
// Given a centered object cloud scaled by the model's kappa factor, produce
// the six encoder inputs: feat, coord, grid_coord, code, order, and inverse.
// Morton (z-order) and Skilling Hilbert curves define the spatial orderings;
// each ordering includes a stable sorting permutation and its inverse.
#pragma once
#include <cstdint>
#include <vector>

namespace graspgenx {

// The four serialization orders the encoder uses, in order:
//   0 = "z", 1 = "z-trans", 2 = "hilbert", 3 = "hilbert-trans".
constexpr int kNumOrders = 4;

struct SerializedInputs {
    int N = 0;
    int depth = 0;                     // serialized_depth = bit_length(max grid coord)
    std::vector<float> feat;           // [N,3] == coord
    std::vector<float> coord;          // [N,3] centered+kappa-scaled points
    std::vector<int32_t> grid_coord;   // [N,3] trunc((coord-min)/grid_size)
    std::vector<int64_t> code;         // [4,N] curve codes
    std::vector<int64_t> order;        // [4,N] argsort(code) per order
    std::vector<int64_t> inverse;      // [4,N] inverse permutation per order
};

// coord: [N*3] row-major float (already centered and kappa-scaled).
SerializedInputs serialize(float const* coord, int N, float grid_size);

} // namespace graspgenx
