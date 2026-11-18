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

#include "serialize.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace graspgenx {
namespace {

// Morton (z-order) key: interleave the low `depth` bits of x,y,z with x in the
// most-significant slot of each 3-bit group.
int64_t mortonKey(uint32_t x, uint32_t y, uint32_t z, int depth)
{
    int64_t key = 0;
    for (int i = 0; i < depth; ++i) {
        key |= (int64_t)((x >> i) & 1u) << (3 * i + 2);
        key |= (int64_t)((y >> i) & 1u) << (3 * i + 1);
        key |= (int64_t)((z >> i) & 1u) << (3 * i + 0);
    }
    return key;
}

// Skilling's 3D Hilbert-curve encoding: Gray-code transform, conversion to
// binary, and most-significant-bit-first packing.
int64_t hilbertKey(uint32_t c0, uint32_t c1, uint32_t c2, int num_bits)
{
    constexpr int D = 3;
    const int L = num_bits * D;
    uint32_t coord[D] = {c0, c1, c2};

    // g[dim][bit], bit 0 = most-significant of the num_bits kept bits.
    uint8_t g[D][16] = {{0}};
    for (int dim = 0; dim < D; ++dim)
        for (int bit = 0; bit < num_bits; ++bit)
            g[dim][bit] = (coord[dim] >> (num_bits - 1 - bit)) & 1u;

    // Skilling transform. Per (bit,dim): XOR the tail of dim-0 with the pivot
    // bit, then conditionally reflect the tails of dim and dim-0.
    for (int bit = 0; bit < num_bits; ++bit) {
        for (int dim = 0; dim < D; ++dim) {
            uint8_t mask = g[dim][bit];
            for (int b = bit + 1; b < num_bits; ++b) {
                g[0][b] ^= mask;
                uint8_t tf = (!mask) & (g[0][b] ^ g[dim][b]); // Bits to exchange between the coordinate axes.
                g[dim][b] ^= tf;
                g[0][b] ^= tf;
            }
        }
    }

    // Interleave to MSB-first bit stream: h[bit*D + dim] = g[dim][bit].
    uint8_t h[48] = {0};
    for (int bit = 0; bit < num_bits; ++bit)
        for (int dim = 0; dim < D; ++dim)
            h[bit * D + dim] = g[dim][bit];

    // Convert Gray code to binary: shift = 2^(ceil(log2(L))-1), halving to 1.
    int cl = 0;
    while ((1 << cl) < L) ++cl;
    for (int shift = 1 << (cl - 1); shift > 0; shift >>= 1)
        for (int i = L - 1; i >= shift; --i)
            h[i] ^= h[i - shift];

    // Pack MSB-first: code = sum_i h[i] << (L-1-i).
    int64_t code = 0;
    for (int i = 0; i < L; ++i)
        if (h[i]) code |= (1LL << (L - 1 - i));
    return code;
}

} // namespace

SerializedInputs serialize(float const* coord, int N, float grid_size)
{
    SerializedInputs out;
    out.N = N;
    out.feat.assign(coord, coord + N * 3);
    out.coord.assign(coord, coord + N * 3);

    // grid_coord = trunc((coord - coord.min(0)) / grid_size), int32.
    float mn[3] = {coord[0], coord[1], coord[2]};
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < 3; ++d)
            mn[d] = std::min(mn[d], coord[i * 3 + d]);
    out.grid_coord.resize(N * 3);
    int32_t maxg = 0;
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < 3; ++d) {
            int32_t v = (int32_t)std::trunc((coord[i * 3 + d] - mn[d]) / grid_size);
            out.grid_coord[i * 3 + d] = v;
            maxg = std::max(maxg, v);
        }

    // depth = bit_length(max grid coord).
    int depth = 0;
    while (maxg >> depth) ++depth;
    out.depth = depth;

    // Curve codes for the four orders.
    out.code.resize(kNumOrders * N);
    for (int i = 0; i < N; ++i) {
        uint32_t x = out.grid_coord[i * 3 + 0];
        uint32_t y = out.grid_coord[i * 3 + 1];
        uint32_t z = out.grid_coord[i * 3 + 2];
        out.code[0 * N + i] = mortonKey(x, y, z, depth);        // z
        out.code[1 * N + i] = mortonKey(y, x, z, depth);        // z-trans
        out.code[2 * N + i] = hilbertKey(x, y, z, depth);       // hilbert
        out.code[3 * N + i] = hilbertKey(y, x, z, depth);       // hilbert-trans
    }

    // order = stable argsort(code) per row; inverse = its inverse permutation.
    out.order.resize(kNumOrders * N);
    out.inverse.resize(kNumOrders * N);
    for (int o = 0; o < kNumOrders; ++o) {
        int64_t const* c = out.code.data() + o * N;
        int64_t* ord = out.order.data() + o * N;
        int64_t* inv = out.inverse.data() + o * N;
        std::iota(ord, ord + N, 0);
        std::stable_sort(ord, ord + N, [c](int64_t a, int64_t b) { return c[a] < c[b]; });
        for (int j = 0; j < N; ++j) inv[ord[j]] = j;
    }
    return out;
}

} // namespace graspgenx
