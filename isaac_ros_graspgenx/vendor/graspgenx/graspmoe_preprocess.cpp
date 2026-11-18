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

#include "graspmoe_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace graspgenx {
namespace {

constexpr double kPi = 3.14159265358979323846;

// ---- 3D KD-tree for exact kNN (median split). metric: 1=L1, 2=L2. ----
// Heap values are the "prune metric" (L1 distance, or SQUARED L2). The mean
// over the k nearest is identical to a brute-force scan; tied points share
// the same distance.
struct KDTree {
    std::vector<std::array<double, 3>> const& P;
    struct Node { int axis, point, left, right; };
    std::vector<Node> nodes;
    int root;

    explicit KDTree(std::vector<std::array<double, 3>> const& pts) : P(pts) {
        std::vector<int> id(P.size());
        for (size_t i = 0; i < P.size(); ++i) id[i] = static_cast<int>(i);
        nodes.reserve(P.size());
        root = build(id, 0, static_cast<int>(id.size()), 0);
    }
    int build(std::vector<int>& id, int lo, int hi, int depth) {
        if (lo >= hi) return -1;
        int axis = depth % 3, mid = (lo + hi) / 2;
        std::nth_element(id.begin() + lo, id.begin() + mid, id.begin() + hi,
                         [&](int a, int b) { return P[a][axis] < P[b][axis]; });
        int node = static_cast<int>(nodes.size());
        nodes.push_back({axis, id[mid], -1, -1});
        int l = build(id, lo, mid, depth + 1);
        int r = build(id, mid + 1, hi, depth + 1);
        nodes[node].left = l; nodes[node].right = r;
        return node;
    }
    static double pd(std::array<double, 3> const& a, std::array<double, 3> const& b,
                     int metric) {
        double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        return metric == 1 ? std::abs(dx) + std::abs(dy) + std::abs(dz)
                           : dx * dx + dy * dy + dz * dz;  // squared for L2
    }
    // mean distance to the k nearest neighbours of point `self` (excluded).
    double mean_knn(int self, int k, int metric) const {
        std::array<double, 3> const& q = P[self];
        std::vector<double> heap;  // max-heap of prune metrics, size <= k
        heap.reserve(k + 1);
        query(root, q, self, k, metric, heap);
        double s = 0.0;
        for (double v : heap) s += (metric == 1) ? v : std::sqrt(v);
        return heap.empty() ? 0.0 : s / heap.size();
    }
    void query(int node, std::array<double, 3> const& q, int self, int k,
               int metric, std::vector<double>& heap) const {
        if (node < 0) return;
        Node const& nd = nodes[node];
        if (nd.point != self) {
            double d = pd(P[nd.point], q, metric);
            if (static_cast<int>(heap.size()) < k) {
                heap.push_back(d); std::push_heap(heap.begin(), heap.end());
            } else if (d < heap.front()) {
                std::pop_heap(heap.begin(), heap.end()); heap.back() = d;
                std::push_heap(heap.begin(), heap.end());
            }
        }
        double diff = q[nd.axis] - P[nd.point][nd.axis];
        int near = diff < 0 ? nd.left : nd.right;
        int far = diff < 0 ? nd.right : nd.left;
        query(near, q, self, k, metric, heap);
        double axisd = metric == 1 ? std::abs(diff) : diff * diff;
        if (static_cast<int>(heap.size()) < k || axisd < heap.front())
            query(far, q, self, k, metric, heap);
    }
};

// Per-point mean distance to the k nearest neighbours (excludes self).
std::vector<double> knn_mean_dist(std::vector<std::array<double, 3>> const& p,
                                  int k, int metric) {
    int const n = static_cast<int>(p.size());
    std::vector<double> out(n, 0.0);
    if (n <= 1) return out;
    KDTree tree(p);
    int kk = std::min(k, n - 1);
#pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; ++i) out[i] = tree.mean_knn(i, kk, metric);
    return out;
}

// Percentile with linear interpolation between sorted samples. q in [0,100].
double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    int n = static_cast<int>(v.size());
    if (n == 1) return v[0];
    double rank = q / 100.0 * (n - 1);
    int lo = static_cast<int>(std::floor(rank));
    int hi = static_cast<int>(std::ceil(rank));
    double frac = rank - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// 2D convex hull (Andrew's monotone chain), CCW, minimal vertices.
std::vector<std::array<double, 2>> convex_hull_2d(
    std::vector<std::array<double, 2>> pts) {
    std::sort(pts.begin(), pts.end(),
              [](auto const& a, auto const& b) {
                  return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
              });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    auto cross = [](std::array<double, 2> const& O, std::array<double, 2> const& A,
                    std::array<double, 2> const& B) {
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
    };
    std::vector<std::array<double, 2>> h(2 * n);
    int kk = 0;
    for (int i = 0; i < n; ++i) {  // lower
        while (kk >= 2 && cross(h[kk - 2], h[kk - 1], pts[i]) <= 0) kk--;
        h[kk++] = pts[i];
    }
    for (int i = n - 2, t = kk + 1; i >= 0; --i) {  // upper
        while (kk >= t && cross(h[kk - 2], h[kk - 1], pts[i]) <= 0) kk--;
        h[kk++] = pts[i];
    }
    h.resize(kk - 1);
    return h;
}

// Minimum-area rectangle angle in radians: evaluate each convex-hull edge
// orientation and keep the rotation with the smallest bounding area.
double min_area_rect_angle(std::vector<std::array<double, 2>> const& xy) {
    auto hull = convex_hull_2d(xy);
    int n = static_cast<int>(hull.size());
    double best_area = std::numeric_limits<double>::infinity(), best_angle = 0.0;
    for (int i = 0; i < n; ++i) {
        auto const& p0 = hull[i];
        auto const& p1 = hull[(i + 1) % n];
        double ex = p1[0] - p0[0], ey = p1[1] - p0[1];
        if (std::sqrt(ex * ex + ey * ey) < 1e-9) continue;
        double angle = std::atan2(ey, ex);
        double c = std::cos(-angle), s = std::sin(-angle);
        double xmin = 1e300, ymin = 1e300, xmax = -1e300, ymax = -1e300;
        for (auto const& q : hull) {
            double rx = c * q[0] - s * q[1];
            double ry = s * q[0] + c * q[1];
            xmin = std::min(xmin, rx); xmax = std::max(xmax, rx);
            ymin = std::min(ymin, ry); ymax = std::max(ymax, ry);
        }
        double area = (xmax - xmin) * (ymax - ymin);
        if (area < best_area) { best_area = area; best_angle = angle; }
    }
    return best_angle;
}

// Percentile extents in the rotated frame. pc is [n,3] double.
Obb obb_from_angle(std::vector<std::array<double, 3>> const& pc, double angle) {
    double c = std::cos(-angle), s = std::sin(-angle);
    // R2d = [[c,-s],[s,c]]; rotated_xy = xy @ R2d.T  (row vec) -> per point:
    //   rx = c*x - s*y ; ry = s*x + c*y
    int n = static_cast<int>(pc.size());
    std::vector<double> rx(n), ry(n), z(n);
    for (int i = 0; i < n; ++i) {
        rx[i] = c * pc[i][0] - s * pc[i][1];
        ry[i] = s * pc[i][0] + c * pc[i][1];
        z[i] = pc[i][2];
    }
    double mins[3] = {percentile(rx, 2.0), percentile(ry, 2.0), percentile(z, 2.0)};
    double maxs[3] = {percentile(rx, 98.0), percentile(ry, 98.0), percentile(z, 98.0)};
    double ext[3], cl[3];
    for (int d = 0; d < 3; ++d) { ext[d] = maxs[d] - mins[d]; cl[d] = (mins[d] + maxs[d]) / 2.0; }
    // R = eye(3); R[:2,:2] = R2d.T = [[c,s],[-s,c]]
    double R[9] = {c, s, 0, -s, c, 0, 0, 0, 1};
    Obb obb;
    for (int r = 0; r < 3; ++r) {
        double v = 0.0;
        for (int cc = 0; cc < 3; ++cc) v += R[r * 3 + cc] * cl[cc];
        obb.center[r] = static_cast<float>(v);
        obb.half_extent[r] = static_cast<float>(ext[r] / 2.0);
    }
    for (int t = 0; t < 9; ++t) obb.R[t] = static_cast<float>(R[t]);
    obb.valid = true;
    return obb;
}

void normalize3(double v[3]) {
    double nrm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    nrm = std::max(nrm, 1e-12);
    for (int i = 0; i < 3; ++i) v[i] /= nrm;
}
void cross3(double const a[3], double const b[3], double o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

std::vector<double> interior_positions(double half, double spacing) {
    if (half <= spacing || spacing <= 0.0) return {0.0};
    std::vector<double> out;
    double eps = 1e-9;
    for (double v = -half + spacing; v < half - spacing + eps; v += spacing)
        out.push_back(v);
    if (out.empty()) out.push_back(0.0);
    return out;
}

// Append row-major 4x4 world-frame grasp poses for one OBB face to `out`.
void build_face_candidates(double const face_origin[3], double const approach[3],
                           double const in_plane[3], std::vector<double> const& positions,
                           std::vector<double> const& yaws,
                           std::vector<double> const& z_offsets, double depth,
                           std::vector<float>& out) {
    double n[3] = {approach[0], approach[1], approach[2]};
    normalize3(n);
    double gz[3] = {-n[0], -n[1], -n[2]};
    double dot = in_plane[0] * gz[0] + in_plane[1] * gz[1] + in_plane[2] * gz[2];
    double gx[3] = {in_plane[0] - dot * gz[0], in_plane[1] - dot * gz[1],
                    in_plane[2] - dot * gz[2]};
    double nrm = std::sqrt(gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2]);
    if (nrm < 1e-9) {
        double fb[3] = {std::abs(n[0]) < 0.9 ? 1.0 : 0.0,
                        std::abs(n[0]) < 0.9 ? 0.0 : 1.0, 0.0};
        double fdot = fb[0] * gz[0] + fb[1] * gz[1] + fb[2] * gz[2];
        for (int i = 0; i < 3; ++i) gx[i] = fb[i] - fdot * gz[i];
        nrm = std::sqrt(gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2]);
    }
    for (int i = 0; i < 3; ++i) gx[i] /= nrm;
    double gy[3];
    cross3(gz, gx, gy);
    // base_R columns [gx gy gz] -> base_R[r][c]
    double baseR[9] = {gx[0], gy[0], gz[0], gx[1], gy[1], gz[1], gx[2], gy[2], gz[2]};

    for (double p : positions) {
        double anchor[3] = {face_origin[0] + p * in_plane[0],
                            face_origin[1] + p * in_plane[1],
                            face_origin[2] + p * in_plane[2]};
        for (double yaw : yaws) {
            double c = std::cos(yaw), s = std::sin(yaw);
            // R_yaw_local = [[c,-s,0],[s,c,0],[0,0,1]]; R_grasp = baseR @ R_yaw
            double Rg[9];
            double Ry[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
            for (int r = 0; r < 3; ++r)
                for (int cc = 0; cc < 3; ++cc) {
                    double v = 0.0;
                    for (int t = 0; t < 3; ++t) v += baseR[r * 3 + t] * Ry[t * 3 + cc];
                    Rg[r * 3 + cc] = v;
                }
            for (double zo : z_offsets) {
                double t3[3] = {anchor[0] + zo * n[0], anchor[1] + zo * n[1],
                                anchor[2] + zo * n[2]};
                // T = [[Rg, t3],[0,0,0,1]]; then T = T @ base_offset where
                // base_offset shifts -depth along local Z: t' = t3 - depth * Rg[:,2]
                double tp[3] = {t3[0] - depth * Rg[2], t3[1] - depth * Rg[5],
                                t3[2] - depth * Rg[8]};
                float m[16] = {(float)Rg[0], (float)Rg[1], (float)Rg[2], (float)tp[0],
                               (float)Rg[3], (float)Rg[4], (float)Rg[5], (float)tp[1],
                               (float)Rg[6], (float)Rg[7], (float)Rg[8], (float)tp[2],
                               0, 0, 0, 1};
                out.insert(out.end(), m, m + 16);
            }
        }
    }
}

}  // namespace

std::vector<uint8_t> sor_keep_mask_l1(float const* pc, int N, float threshold, int k) {
    std::vector<std::array<double, 3>> p(N);
    for (int i = 0; i < N; ++i)
        p[i] = {pc[i * 3 + 0], pc[i * 3 + 1], pc[i * 3 + 2]};
    auto mean = knn_mean_dist(p, k, /*metric=*/1);
    std::vector<uint8_t> keep(N);
    for (int i = 0; i < N; ++i) keep[i] = (mean[i] < threshold) ? 1 : 0;
    return keep;
}

Obb compute_obb(float const* pc, int N, MoeParams const& m) {
    std::vector<std::array<double, 3>> pts(N);
    std::mt19937 gen(0);
    std::normal_distribution<double> jitter(0.0, 1e-4);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < 3; ++d)
            pts[i][d] = pc[i * 3 + d] + (m.obb_jitter ? jitter(gen) : 0.0);

    // internal SOR (L2, std_ratio=2.0)
    auto mean = knn_mean_dist(pts, m.outlier_k, /*metric=*/2);
    int n = N;
    double mu = std::accumulate(mean.begin(), mean.end(), 0.0) / n;
    double var = 0.0;
    for (double v : mean) var += (v - mu) * (v - mu);
    double sd = std::sqrt(var / n);  // Population standard deviation (divide variance by n).
    double cut = mu + 2.0 * sd;
    std::vector<std::array<double, 3>> clean;
    clean.reserve(n);
    for (int i = 0; i < n; ++i)
        if (mean[i] < cut) clean.push_back(pts[i]);

    Obb obb;
    if (static_cast<int>(clean.size()) < 4) { obb.valid = false; return obb; }
    std::vector<std::array<double, 2>> xy(clean.size());
    for (size_t i = 0; i < clean.size(); ++i) xy[i] = {clean[i][0], clean[i][1]};
    double angle = min_area_rect_angle(xy);
    return obb_from_angle(clean, angle);
}

std::vector<float> obb_candidates(Obb const& obb, GripperParams const& g,
                                  MoeParams const& m, bool& skipped) {
    skipped = false;
    std::vector<float> out;
    if (!obb.valid) { skipped = true; return out; }

    double he[3] = {obb.half_extent[0], obb.half_extent[1], obb.half_extent[2]};
    // auto skip rule: full extents all > gripper width
    if (he[0] * 2 > g.width_m && he[1] * 2 > g.width_m && he[2] * 2 > g.width_m) {
        skipped = true;
        return out;
    }
    double center[3] = {obb.center[0], obb.center[1], obb.center[2]};
    // R columns: x_axis = R[:,0], y_axis = R[:,1]
    double xax[3] = {obb.R[0], obb.R[3], obb.R[6]};
    double yax[3] = {obb.R[1], obb.R[4], obb.R[7]};
    double world_z_half = std::abs(obb.R[6]) * he[0] + std::abs(obb.R[7]) * he[1] +
                          std::abs(obb.R[8]) * he[2];
    double top_z = center[2] + world_z_half;

    std::vector<double> yaws(m.num_yaws);
    for (int i = 0; i < m.num_yaws; ++i) yaws[i] = 2.0 * kPi * i / m.num_yaws;
    std::vector<double> zoff(m.z_offsets_m.begin(), m.z_offsets_m.end());
    double sp = m.obb_position_spacing_m, depth = g.depth_m;
    double zworld[3] = {0, 0, 1};

    // top face (approach = +z_world; in-plane = OBB long horizontal axis)
    int top_long = (he[0] >= he[1]) ? 0 : 1;
    auto top_pos = interior_positions(he[top_long], sp);
    double top_inplane[3] = {obb.R[top_long], obb.R[3 + top_long], obb.R[6 + top_long]};
    double top_origin[3] = {center[0], center[1], top_z};
    build_face_candidates(top_origin, zworld, top_inplane, top_pos, yaws, zoff, depth, out);

    // +/- x faces (in-plane = y_axis; interior positions along half_extent[1])
    auto pos_x = interior_positions(he[1], sp);
    for (double sign : {+1.0, -1.0}) {
        double fo[3] = {center[0] + sign * he[0] * xax[0],
                        center[1] + sign * he[0] * xax[1], center[2]};
        double appr[3] = {sign * xax[0], sign * xax[1], sign * xax[2]};
        build_face_candidates(fo, appr, yax, pos_x, yaws, zoff, depth, out);
    }
    // +/- y faces (in-plane = x_axis; interior positions along half_extent[0])
    auto pos_y = interior_positions(he[0], sp);
    for (double sign : {+1.0, -1.0}) {
        double fo[3] = {center[0] + sign * he[1] * yax[0],
                        center[1] + sign * he[1] * yax[1], center[2]};
        double appr[3] = {sign * yax[0], sign * yax[1], sign * yax[2]};
        build_face_candidates(fo, appr, xax, pos_y, yaws, zoff, depth, out);
    }
    return out;
}

PreprocOut preprocess(float const* raw_pc, int N, GripperParams const& g,
                      MoeParams const& m) {
    PreprocOut out;
    auto keep = sor_keep_mask_l1(raw_pc, N, m.outlier_threshold, m.outlier_k);
    for (int i = 0; i < N; ++i)
        if (keep[i])
            out.filtered_pc.insert(out.filtered_pc.end(), raw_pc + i * 3, raw_pc + i * 3 + 3);
    int M = static_cast<int>(out.filtered_pc.size() / 3);
    if (M < 4) return out;
    out.obb = compute_obb(out.filtered_pc.data(), M, m);
    out.candidates = obb_candidates(out.obb, g, m, out.skipped_obb);
    return out;
}

}  // namespace graspgenx
