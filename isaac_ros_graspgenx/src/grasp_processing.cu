// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_graspgenx/grasp_processing.hpp"

#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cmath>

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

// Read one possibly padded PointCloud2 record per thread and transform XYZ directly into the
// origin frame. Keeping the layout metadata separate avoids repacking the CUDA-backed cloud.
__global__ void ReadAndTransformPoints(
  const uint8_t * cloud, PointCloudInputMetadata cloud_metadata, float * points)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cloud_metadata.count) {return;}
  // Convert the flat point index into an organized-cloud row and column, then apply both row and
  // point byte strides. This also handles unorganized clouds, whose height is one.
  const auto * source = cloud + static_cast<size_t>
    (i / cloud_metadata.width) * cloud_metadata.row_step +
    static_cast<size_t>(i % cloud_metadata.width) * cloud_metadata.point_step;

  // PointCloud2 fields need not be contiguous or ordered, so read XYZ through their byte offsets.
  float xyz[3];
  for (int d = 0; d < 3; ++d) {
    xyz[d] = *reinterpret_cast<const float *>(source + cloud_metadata.xyz_offset[d]);
  }
  // Multiply [x, y, z, 1] by the corresponding row of origin_pose_point_cloud.
  for (int d = 0; d < 3; ++d) {
    const float * row = cloud_metadata.origin_pose_point_cloud + d * 4;
    points[i * 3 + d] = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2] + row[3];
  }
}

// Exact L1 k-nearest-neighbor threshold, excluding self. One thread scans the full cloud for one
// point, giving O(count^2) work. This could be optimized for large clouds with a GPU k-d tree
// while preserving the exact L1 nearest-neighbor semantics.
__global__ void FilterPoints(
  const float * points, int count, int k, float threshold,
  int * keep)
{
  // Assign one input point to each CUDA thread.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {return;}

  // Reject the point by default so every early return leaves a valid keep-mask entry.
  const float * point = points + i * 3;
  keep[i] = 0;
  if (!isfinite(point[0]) || !isfinite(point[1]) || !isfinite(point[2])) {return;}

  // Keep a sorted, fixed-capacity list of this point's nearest L1 distances.
  double nearest[kMaxNeighbors];
  for (int n = 0; n < k; ++n) {
    nearest[n] = INFINITY;
  }

  int neighbors = 0;
  for (int j = 0; j < count; ++j) {
    // A point is not its own neighbor.
    if (i == j) {continue;}

    const float * other = points + j * 3;
    const double distance = fabs(static_cast<double>(point[0]) - other[0]) +
      fabs(static_cast<double>(point[1]) - other[1]) +
      fabs(static_cast<double>(point[2]) - other[2]);
    // A non-finite coordinate in the other point makes its distance non-finite.
    if (!isfinite(distance)) {continue;}
    ++neighbors;

    // Insert only distances that belong in the nearest-k list. k is capped by kMaxNeighbors, so
    // insertion into this small per-thread array avoids dynamic temporary storage.
    if (distance >= nearest[k - 1]) {continue;}
    int position = k - 1;
    while (position > 0 && distance < nearest[position - 1]) {
      nearest[position] = nearest[position - 1];
      --position;
    }
    nearest[position] = distance;
  }

  // Small clouds may contain fewer than k finite neighbors, so average only those available.
  const int used = min(k, neighbors);
  double sum = 0.0;
  for (int n = 0; n < used; ++n) {
    sum += nearest[n];
  }
  // Use a strict threshold: equality is rejected.
  keep[i] = used > 0 && sum / used < threshold;
}

// Compact the keep mask in original point order. This serial kernel is intentional: compaction is
// a small setup step relative to the all-pairs filter, and preserving order makes seeded sampling
// reproducible. For large clouds, Thrust or CUB could parallelize this stable compaction.
// status[0] receives the kept count; preprocessing error/output fields are reset.
__global__ void CompactIndices(int count, ProcessingWorkspace workspace)
{
  // This kernel is launched as <<<1, 1>>>, so one thread can append without atomics.
  int selected = 0;
  for (int i = 0; i < count; ++i) {
    // Scanning from zero upward preserves the original point order.
    if (workspace.keep[i]) {workspace.indices[selected++] = i;}
  }

  // Publish the number of valid entries in indices and clear later-stage status fields.
  workspace.status[0] = selected;
  workspace.status[1] = 0;
  workspace.status[2] = 0;
}

// Draw the engine's fixed 3,500 points with replacement from the filtered index list. Assigning
// one Philox subsequence to each output position makes the result deterministic for a given seed.
__global__ void SamplePoints(ProcessingWorkspace workspace, uint64_t seed, float * coord)
{
  // Each thread produces one row of the fixed [kModelPoints, 3] output tensor.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kModelPoints) {return;}

  // Use the output row as the Philox subsequence, giving every row an independent deterministic
  // random stream without sharing mutable generator state between threads.
  curandStatePhilox4_32_10_t random;
  curand_init(seed, i, 0, &random);

  // status[0] is the number of valid entries written to workspace.indices by CompactIndices.
  const int count = workspace.status[0];
  // Sampling is with replacement, so this works whether count is smaller or larger than
  // kModelPoints. curand_uniform_double can return 1.0, and min clamps that endpoint to count - 1.
  const int selected = count > 0 ? workspace.indices[min(
        static_cast<int>(curand_uniform_double(&random) * count), count - 1)] : 0;

  // If filtering retained no points, write zeros without dereferencing an invalid index. The host
  // rejects this request after preprocessing because fewer than four points survived.
  for (int d = 0; d < 3; ++d) {
    coord[i * 3 + d] = count > 0 ? workspace.points[selected * 3 + d] : 0.0F;
  }
}

// Save the sampled centroid in origin-frame meters. Serial accumulation preserves the
// summation order so seeded inference remains deterministic across launches.
__global__ void ComputeCenter(const float * coord, float * center)
{
  // Compute and retain the sampled centroid before normalization. Decode uses this value to move
  // model translations back into the origin frame.
  for (int d = 0; d < 3; ++d) {
    float sum = 0.0F;
    for (int i = 0; i < kModelPoints; ++i) {
      sum += coord[i * 3 + d];
    }
    center[d] = sum / kModelPoints;
  }
}

// Normalize one point per thread after ComputeCenter completes on the same stream.
__global__ void CenterAndScale(float * coord, float kappa, const float * center)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kModelPoints) {return;}
  // The engine was trained on zero-centered coordinates multiplied by the fixed model scale.
  for (int d = 0; d < 3; ++d) {
    coord[i * 3 + d] = (coord[i * 3 + d] - center[d]) * kappa;
  }
}

// The same Morton and Skilling Hilbert encodings as vendor/graspgenx/serialize.cpp.
__device__ int64_t MortonKey(uint32_t x, uint32_t y, uint32_t z, int depth)
{
  int64_t key = 0;
  // Interleave one bit from each axis into each three-bit group: [x_bit, y_bit, z_bit].
  for (int bit = 0; bit < depth; ++bit) {
    key |= static_cast<int64_t>((x >> bit) & 1U) << (3 * bit + 2);
    key |= static_cast<int64_t>((y >> bit) & 1U) << (3 * bit + 1);
    key |= static_cast<int64_t>((z >> bit) & 1U) << (3 * bit);
  }
  return key;
}

__device__ int64_t HilbertKey(uint32_t x, uint32_t y, uint32_t z, int depth)
{
  if (depth == 0) {return 0;}
  const uint32_t coord[3] = {x, y, z};
  uint8_t g[3][16] {};
  // Expand each coordinate into a most-significant-bit-first array. depth is at most 16 because
  // ComputeGridCoordinates rejects coordinates outside the unsigned 16-bit range.
  for (int d = 0; d < 3; ++d) {
    for (int bit = 0; bit < depth; ++bit) {
      g[d][bit] = (coord[d] >> (depth - 1 - bit)) & 1U;
    }
  }
  // Apply Skilling's transpose transform, which reflects and exchanges coordinate tails according
  // to each pivot bit so adjacent Hilbert cells receive adjacent Gray-code states.
  for (int bit = 0; bit < depth; ++bit) {
    for (int d = 0; d < 3; ++d) {
      const uint8_t mask = g[d][bit];
      for (int b = bit + 1; b < depth; ++b) {
        g[0][b] ^= mask;
        const uint8_t flip = (!mask) & (g[0][b] ^ g[d][b]);
        g[d][b] ^= flip;
        g[0][b] ^= flip;
      }
    }
  }
  uint8_t bits[48] {};
  // Interleave the transformed axis bits into one 3 * depth bit stream.
  for (int bit = 0; bit < depth; ++bit) {
    for (int d = 0; d < 3; ++d) {
      bits[bit * 3 + d] = g[d][bit];
    }
  }
  const int length = depth * 3;
  int shift = 1;
  while (shift < length) {shift <<= 1;}
  // Convert the interleaved Gray-code stream to binary with a parallel-prefix XOR.
  for (shift >>= 1; shift > 0; shift >>= 1) {
    for (int i = length - 1; i >= shift; --i) {
      bits[i] ^= bits[i - shift];
    }
  }
  int64_t code = 0;
  // Pack the binary stream most-significant bit first into the final spatial key.
  for (int i = 0; i < length; ++i) {
    code |= static_cast<int64_t>(bits[i]) << (length - 1 - i);
  }
  return code;
}

// Quantize centered coordinates into nonnegative grid cells. Subtracting the per-axis minimum
// makes truncation equivalent to floor. A 16-bit-per-axis limit keeps Hilbert codes within the
// fixed 48-bit scratch representation used above. status[1] reports invalid cells and status[3]
// records the number of significant bits needed by MortonKey and HilbertKey.
__global__ void ComputeGridCoordinates(
  TensorRtInputTensors tensors, float grid_size, int * status)
{
  int maximum = 0;
  // Find the lower corner of the sampled cloud's axis-aligned bounding box. Using this as the
  // origin guarantees nonnegative grid coordinates without changing relative cell positions.
  float minimum[3] = {INFINITY, INFINITY, INFINITY};
  for (int i = 0; i < kModelPoints; ++i) {
    for (int d = 0; d < 3; ++d) {
      minimum[d] = fminf(minimum[d], tensors.coord[i * 3 + d]);
    }
  }

  for (int i = 0; i < kModelPoints; ++i) {
    for (int d = 0; d < 3; ++d) {
      // Convert the offset from the lower corner into a zero-based integer cell coordinate.
      const float value = (tensors.coord[i * 3 + d] - minimum[d]) / grid_size;
      // Morton/Hilbert storage supports at most 16 bits per axis. Record an error instead of
      // allowing an invalid coordinate to overflow or index undefined encoding state.
      const bool valid = isfinite(value) && value >= 0.0F && value < 65536.0F;
      if (!valid) {status[1] = 1;}
      tensors.grid_coord[i * 3 + d] = valid ? static_cast<int32_t>(value) : 0;
      // The largest cell determines how many low bits each spatial encoder must consume.
      maximum = max(maximum, tensors.grid_coord[i * 3 + d]);
    }
  }
  // __clz counts leading zeros, so 32 - __clz(maximum) is maximum's integer bit length.
  // __clz(0) is undefined, hence the explicit all-zero case.
  status[3] = maximum == 0 ? 0 : 32 - __clz(maximum);
}

// Encode each point in the four spatial orders expected by the engine. Rows are laid out as
// Morton XYZ, Morton YXZ, Hilbert XYZ, and Hilbert YXZ; points remain in sampled input order.
__global__ void MakeCodes(TensorRtInputTensors tensors, const int * status)
{
  // Assign one sampled point to each thread.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kModelPoints) {return;}

  // ComputeGridCoordinates stores the shared per-axis bit depth in status[3].
  const int depth = status[3];
  const uint32_t x = tensors.grid_coord[i * 3];
  const uint32_t y = tensors.grid_coord[i * 3 + 1];
  const uint32_t z = tensors.grid_coord[i * 3 + 2];

  // Each contiguous row contains one complete spatial ordering. Swapping X and Y gives the model a
  // second orientation for each curve without changing the underlying sampled point indices.
  tensors.code[i] = MortonKey(x, y, z, depth);
  tensors.code[kModelPoints + i] = MortonKey(y, x, z, depth);
  tensors.code[2 * kModelPoints + i] = HilbertKey(x, y, z, depth);
  tensors.code[3 * kModelPoints + i] = HilbertKey(y, x, z, depth);
}

// Sort point indices by an external code row without moving or modifying the codes themselves.
struct CompareCode
{
  const int64_t * code;
  __device__ bool operator()(int64_t left, int64_t right) const
  {
    return code[left] < code[right];
  }
};

// Convert each sorted-index permutation into the reverse lookup used by the encoder:
// inverse[original_point] = position_in_spatial_order.
__global__ void InvertOrders(TensorRtInputTensors tensors)
{
  // Flatten the [order, point] launch space so every permutation entry is handled once.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kModelPoints * kSpatialOrders) {return;}
  const int row = i / kModelPoints;
  // order[i] is an original point index; store its column in this sorted row.
  tensors.inverse[row * kModelPoints + tensors.order[i]] = i % kModelPoints;
}

// Generate both diffusion-noise tensors in one launch. Offsetting Philox subsequences by
// kModelPoints keeps these streams disjoint from the subsequences used for point sampling.
__global__ void MakeNoise(TensorRtInputTensors tensors, uint64_t seed)
{
  // Flatten initial noise followed by all per-step noise into one launch index range.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int initial_count = kModelGrasps * 6;
  if (i >= initial_count * (kDiffusionSteps + 1)) {return;}

  // Each scalar receives an independent deterministic Philox subsequence.
  curandStatePhilox4_32_10_t random;
  curand_init(seed, kModelPoints + i, 0, &random);
  const float noise = curand_normal(&random);
  // The first [kModelGrasps, 6] block initializes candidates; the remaining contiguous blocks
  // form [kDiffusionSteps, kModelGrasps, 6].
  if (i < initial_count) {tensors.initial_noise[i] = noise;} else {
    tensors.step_noise[i - initial_count] = noise;
  }
}

__device__ float Determinant(const float * r)
{
  return r[0] * (r[4] * r[8] - r[5] * r[7]) -
         r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
}

// Validate and rank all 200 predictions serially. The candidate count is small, and one thread
// gives deterministic stable ordering without global synchronization or a temporary sort buffer.
__global__ void Decode(
  const float * matrices, const float * confidences, const float * center,
  const float * tool, float threshold, int limit, DeviceGrasp * output, int * status)
{
  int count = 0;
  for (int i = 0; i < kModelGrasps; ++i) {
    // A negative threshold disables score filtering, but non-finite scores are always invalid.
    const float confidence = confidences[i];
    if (!isfinite(confidence) || (threshold >= 0.0F && confidence < threshold)) {continue;}
    const float * matrix = matrices + i * 16;
    // Validate all 16 values before using any matrix element in subsequent arithmetic.
    bool valid = true;
    for (int j = 0; j < 16; ++j) {
      valid = valid && isfinite(matrix[j]);
    }
    // A rigid homogeneous transform must have a bottom row close to [0, 0, 0, 1].
    if (!valid || fabsf(matrix[12]) > 0.05F || fabsf(matrix[13]) > 0.05F ||
      fabsf(matrix[14]) > 0.05F || fabsf(matrix[15] - 1.0F) > 0.05F) {continue;}
    float r[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        r[row * 3 + col] = matrix[row * 4 + col];
      }
    }
    float error = 0.0F;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        float value = row == col ? -1.0F : 0.0F;
        for (int k = 0; k < 3; ++k) {
          value += r[k * 3 + row] * r[k * 3 + col];
        }
        error += value * value;
      }
    }
    // Reject matrices far from SO(3): error is ||R^T R - I||_F^2, and determinant +1 excludes
    // reflections. Small numerical errors are repaired below rather than discarding the grasp.
    if (error > 0.01F || fabsf(Determinant(r) - 1.0F) > 0.1F) {continue;}
    // Newton's polar iteration R <- 0.5 * (R + R^-T) projects a near-rotation onto SO(3).
    // The cofactor divided by determinant is one element of R^-T.
    for (int iteration = 0; iteration < 5; ++iteration) {
      float next[9];
      const float det = Determinant(r);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          const int a = (row + 1) % 3, b = (row + 2) % 3;
          const int c = (col + 1) % 3, d = (col + 2) % 3;
          const float cofactor = r[a * 3 + c] * r[b * 3 + d] - r[a * 3 + d] * r[b * 3 + c];
          next[row * 3 + col] = 0.5F * (r[row * 3 + col] + cofactor / det);
        }
      }
      for (int j = 0; j < 9; ++j) {
        r[j] = next[j];
      }
    }
    DeviceGrasp grasp {};
    grasp.confidence = confidence;
    grasp.pose[15] = 1.0F;
    // Restore the centroid removed before inference, then compose grasp_pose_tool on the right:
    // [R, t + center] * [R_tool, t_tool].
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 4; ++col) {
        float value = col == 3 ? matrix[row * 4 + 3] + center[row] : 0.0F;
        for (int k = 0; k < 3; ++k) {
          value += r[row * 3 + k] * tool[k * 4 + col];
        }
        grasp.pose[row * 4 + col] = value;
      }
    }
    // Insert into the bounded descending-confidence list. Strict comparison retains model input
    // order when scores are equal.
    int position = count;
    while (position > 0 && confidence > output[position - 1].confidence) {--position;}
    if (position >= limit) {continue;}
    for (int j = min(count, limit - 1); j > position; --j) {
      output[j] = output[j - 1];
    }
    output[position] = grasp;
    count = min(count + 1, limit);
  }
  status[2] = count;
}

}  // namespace

void SerializePoints(
  TensorRtInputTensors tensors, float grid_size, int * status,
  cudaStream_t stream)
{
  // Grid bounds and bit depth require reductions over all points, so one thread computes them
  // before the independently parallel code generation.
  ComputeGridCoordinates <<<1, 1, 0, stream>>>(tensors, grid_size, status);
  MakeCodes <<<(kModelPoints + 127) / 128, 128, 0, stream>>>(tensors, status);
  const auto policy = thrust::cuda::par.on(stream);
  for (int row = 0; row < kSpatialOrders; ++row) {
    auto order = thrust::device_pointer_cast(tensors.order + row * kModelPoints);
    thrust::sequence(policy, order, order + kModelPoints, int64_t {0});
    // Stable sorting gives points in the same grid cell a deterministic original-order tie break.
    thrust::stable_sort(
      policy, order, order + kModelPoints,
      CompareCode {tensors.code + row * kModelPoints});
  }
  InvertOrders <<<(kModelPoints * kSpatialOrders + 127) / 128, 128, 0, stream>>>(tensors);
}

void PrepareModelInputs(
  const uint8_t * cloud, PointCloudInputMetadata cloud_metadata,
  PointCloudPreprocessingSettings settings,
  ProcessingWorkspace workspace, TensorRtInputTensors tensors, cudaStream_t stream)
{
  const int blocks = (cloud_metadata.count + 127) / 128;
  // Convert the borrowed cloud into origin-frame XYZ and reject non-finite/spatial outliers.
  ReadAndTransformPoints <<<blocks, 128, 0, stream>>>(
    cloud, cloud_metadata, workspace.points);
  FilterPoints <<<blocks, 128, 0, stream>>>(
    workspace.points, cloud_metadata.count, settings.neighbors, settings.outlier_threshold,
    workspace.keep);
  // Build a stable retained-index list, then sample the engine's fixed point count from it.
  CompactIndices <<<1, 1, 0, stream>>>(cloud_metadata.count, workspace);
  SamplePoints <<<(kModelPoints + 127) / 128, 128, 0,
    stream>>>(workspace, settings.seed, tensors.coord);
  // Produce the normalized geometric tensors and deterministic diffusion inputs expected by TRT.
  ComputeCenter <<<1, 1, 0, stream>>>(tensors.coord, workspace.center);
  CenterAndScale <<<(kModelPoints + 127) / 128, 128, 0, stream>>>(
    tensors.coord, settings.kappa, workspace.center);
  SerializePoints(tensors, settings.grid_size, workspace.status, stream);
  // The diffusion model represents each grasp with six pose parameters, each requiring noise.
  constexpr int noise_count = kModelGrasps * 6 * (kDiffusionSteps + 1);
  MakeNoise <<<(noise_count + 127) / 128, 128, 0, stream>>>(tensors, settings.seed);
}

void DecodeGrasps(
  const float * matrices, const float * confidences, const float * center,
  const float * grasp_pose_tool, float threshold, int limit,
  DeviceGrasp * result, int * status, cudaStream_t stream)
{
  Decode <<<1, 1, 0, stream>>>(
    matrices, confidences, center, grasp_pose_tool, threshold, limit, result, status);
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia
