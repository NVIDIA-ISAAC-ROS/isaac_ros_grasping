// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_GRASPGENX__GRASP_PROCESSING_HPP_
#define ISAAC_ROS_GRASPGENX__GRASP_PROCESSING_HPP_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

// Number of sampled object points required by the engine's fixed input tensor shapes.
constexpr int kModelPoints = 3500;

// Number of grasp candidates produced by the engine before filtering and ranking.
constexpr int kModelGrasps = 200;

// Number of diffusion denoising steps in the engine; sizes the per-step noise tensor.
constexpr int kDiffusionSteps = 20;

// Number of point orderings: Morton and Hilbert, each with XYZ and YXZ axis order.
constexpr int kSpatialOrders = 4;

// Maximum neighbor count supported by the outlier filter's per-point distance array.
constexpr int kMaxNeighbors = 32;

// Host metadata only. The point payload and every workspace/tensor pointer below are on the GPU.
struct PointCloudInputMetadata
{
  // Total number of input points, including invalid points; equals cloud width * height.
  int count;

  // Number of points per row, used to convert a point index into row and column indices.
  uint32_t width;

  // Byte stride between consecutive points within a row, including any padding fields.
  uint32_t point_step;

  // Byte stride between consecutive rows, including any row padding.
  uint32_t row_step;

  // Byte offsets of the aligned float32 x, y, and z fields within each point record.
  uint32_t xyz_offset[3];

  // Row-major 3x4 rigid transform [R | t] mapping cloud coordinates into the origin frame.
  // Point coordinates and the translation are in meters.
  float origin_pose_point_cloud[12];
};

// Host parameters controlling point filtering, model normalization, and random sampling.
struct PointCloudPreprocessingSettings
{
  // Positive, finite scale factor applied after subtracting the sampled cloud's center.
  float kappa;

  // Positive, finite grid cell size in kappa-scaled coordinates, used for spatial ordering.
  float grid_size;

  // Positive, finite upper bound in meters on mean L1 distance to neighboring points.
  // Points whose mean distance reaches or exceeds this value are discarded.
  float outlier_threshold;

  // Number of nearest finite neighbors to average, excluding self; in [1, kMaxNeighbors].
  // Uses all available finite neighbors when fewer than this number remain.
  int neighbors;

  // Philox seed shared by point sampling and diffusion noise generation for repeatable inputs.
  uint64_t seed;
};

// Borrowed GPU buffers filled by preprocessing and consumed by TensorRT. Arrays use row-major
// layout and must remain allocated until all work on the launch stream completes.
struct TensorRtInputTensors
{
  // [kModelPoints, 3] centered, kappa-scaled XYZ values. Also bound as the encoder's feat input.
  float * coord;

  // [kModelPoints, 3] nonnegative integer cells from (coord - per-axis minimum) / grid_size.
  int32_t * grid_coord;

  // [kSpatialOrders, kModelPoints] spatial keys for Morton XYZ, Morton YXZ, Hilbert XYZ,
  // and Hilbert YXZ, respectively. Each row retains the original sampled point order.
  int64_t * code;

  // [kSpatialOrders, kModelPoints] point indices sorted by each row's spatial keys.
  // Equal keys preserve the original sampled point order.
  int64_t * order;

  // [kSpatialOrders, kModelPoints] inverse permutations mapping each point to its sorted position.
  int64_t * inverse;

  // [kModelGrasps, 6] standard normal noise initializing each candidate's six pose components.
  float * initial_noise;

  // [kDiffusionSteps, kModelGrasps, 6] standard normal noise supplied at each diffusion step.
  float * step_noise;
};

// Borrowed GPU scratch buffers carrying intermediate results between preprocessing kernels and
// into grasp decoding. They hold the full input cloud, filtering/sampling state, and the center
// needed to restore predicted translations; these values are separate from the engine inputs.
// The caller allocates these buffers and keeps them valid until all stream work completes.
struct ProcessingWorkspace
{
  // [PointCloudInputMetadata::count, 3] input XYZ values transformed into the origin frame,
  // in meters.
  float * points;

  // [PointCloudInputMetadata::count] flags: 1 for finite points passing the outlier filter,
  // otherwise 0.
  int * keep;

  // [PointCloudInputMetadata::count] storage for kept point indices in input order. Only the first
  // status[0] entries are valid; sampling draws from this list with replacement.
  int * indices;

  // Three-float mean of the sampled points in origin-frame meters, saved before kappa scaling.
  // Decoding adds this center back to the model's predicted translations.
  float * center;

  // Four integers: [0] kept point count; [1] nonzero if grid coordinates are invalid or exceed
  // 16 bits; [2] returned grasp count; [3] bits per grid axis used to encode spatial keys.
  int * status;
};

// One decoded grasp stored on the GPU before selected results are copied to the host.
struct DeviceGrasp
{
  // Row-major 4x4 rigid pose of the tool in the origin frame, with translation in meters.
  float pose[16];

  // Finite model confidence score used to filter and rank this grasp; higher scores rank first.
  float confidence;
};

//! Transform, filter, sample, and normalize an object cloud, then build spatial orders and noise.
//!
//! `cloud` is the borrowed device byte buffer described by the validated `cloud_metadata`.
//! `settings` supplies the filtering, scaling, grid, and seed parameters. Sampling draws
//! kModelPoints points with replacement; sampling and noise use deterministic Philox streams.
//! `workspace` provides scratch storage sized for cloud_metadata.count points and receives the
//! sampled center and status. `tensors` supplies the device allocations receiving all model inputs.
//! The caller must check workspace.status[0] >= 4 and workspace.status[1] == 0 before inference.
//! `stream` is used for all kernels and sorting; null selects the default CUDA stream. Keep all
//! device buffers alive until that work completes and synchronize before reading results on the
//! host. The caller is responsible for checking CUDA launch and execution errors.
void PrepareModelInputs(
  const uint8_t * cloud, PointCloudInputMetadata cloud_metadata,
  PointCloudPreprocessingSettings settings,
  ProcessingWorkspace workspace, TensorRtInputTensors tensors, cudaStream_t stream);

//! Compute grid coordinates, spatial keys, sorted point indices, and inverse permutations.
//!
//! `tensors.coord` contains kModelPoints centered, kappa-scaled XYZ triples. The function fills
//! tensors.grid_coord, code, order, and inverse; the noise pointers are unused.
//! `grid_size` is a positive, finite cell size in the same scaled coordinates.
//! `status` points to four device integers. The caller clears status[1] before calling; the
//! function sets it to 1 for invalid or out-of-range grid coordinates and writes the bit depth
//! to status[3]. It preserves status[0] and status[2].
//! All work uses `stream` (null selects the default stream). Keep buffers alive until completion
//! and synchronize before host reads. This entry point also supports comparison with the
//! upstream CPU serialization reference.
void SerializePoints(
  TensorRtInputTensors tensors, float grid_size, int * status, cudaStream_t stream);

//! Validate model predictions, repair small rotation errors, and retain the highest scoring poses.
//!
//! `matrices` contains kModelGrasps row-major 4x4 model poses with centered translations in meters;
//! `confidences` contains one score per pose. Non-finite scores and invalid poses are discarded.
//! `center` is the three-float origin-frame offset saved during preprocessing.
//! `grasp_pose_tool` is a row-major 4x4 rigid transform composed on the right of each restored grasp
//! pose to obtain the tool pose; its translation is in meters. All pointer arguments are on GPU.
//! `threshold` is the finite minimum accepted score; a negative value disables score filtering.
//! `limit` is in [0, kModelGrasps] and bounds the number of results; zero returns no grasps.
//! `result` has capacity for limit DeviceGrasp entries, ranked by descending confidence with
//! input order preserved for equal scores. `status` points to four device integers; only
//! status[2] is overwritten with the returned count.
//! The kernel runs on `stream` (null selects the default stream). Keep buffers alive until
//! completion and synchronize before host reads; check CUDA launch and execution errors.
void DecodeGrasps(
  const float * matrices, const float * confidences, const float * center,
  const float * grasp_pose_tool, float threshold, int limit,
  DeviceGrasp * result, int * status, cudaStream_t stream);

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GRASPGENX__GRASP_PROCESSING_HPP_
