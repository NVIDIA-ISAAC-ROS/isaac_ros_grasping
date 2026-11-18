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

#ifndef ISAAC_ROS_GRASPGENX__GRASP_INFERENCE_HPP_
#define ISAAC_ROS_GRASPGENX__GRASP_INFERENCE_HPP_

#include <Eigen/Geometry>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

// Describes one TensorRT inference operation: the object cloud and frame transforms, selected
// gripper, candidate filtering, and repeatability controls.
struct InferenceRequest
{
  // Object points in the cloud frame, with XYZ in meters.
  sensor_msgs::msg::PointCloud2::ConstSharedPtr object_cloud;

  // Rigid transform mapping object_cloud coordinates into the frame used for returned poses.
  Eigen::Isometry3d origin_pose_point_cloud{Eigen::Isometry3d::Identity()};

  // Fixed transform composed on the right of a predicted grasp pose to obtain the tool pose.
  // The translation is in meters; identity returns the model's grasp frame without an offset.
  Eigen::Isometry3d grasp_pose_tool{Eigen::Isometry3d::Identity()};

  // Name selecting the gripper geometry configured in the backend for model conditioning.
  std::string gripper_name;

  // Finite minimum accepted confidence score; any negative value disables score filtering.
  float grasp_confidence_threshold{-1.0F};

  // Maximum number of returned grasps before the top_k_grasps cap. The model always evaluates
  // 200 candidates regardless of this value.
  int32_t num_grasps{50};

  // Additional cap on the highest scoring results: -1 uses num_grasps, 0 returns none, and
  // positive values return at most min(num_grasps, top_k_grasps).
  int32_t top_k_grasps{20};

  // Seed controlling repeatable point sampling and diffusion noise generation.
  int64_t random_seed{0};
};

enum class GraspSource
{
  kDiffusion
};

// One decoded model prediction expressed as a rigid grasp pose and confidence score.
struct DecodedGrasp
{
  // Tool pose in the origin frame after applying grasp_pose_tool; translation is in meters.
  Eigen::Isometry3d grasp_pose{Eigen::Isometry3d::Identity()};

  // Model score for this candidate; higher values indicate more confident predictions.
  float confidence{0.0F};

  // Candidate generation method.
  GraspSource source{GraspSource::kDiffusion};
};

// Complete inference response; grasps are valid only on success.
struct InferenceResult
{
  // True when inference completed and at least one valid grasp is available.
  bool success{false};

  // Human-readable result summary or diagnostic explaining why inference failed.
  std::string message;

  // Selected tool poses and scores in descending confidence order; empty on failure.
  std::vector<DecodedGrasp> grasps;
};

inline constexpr float kDefaultKappa = 3.27F;
inline constexpr float kDefaultGridSizeM = 0.01F;
inline constexpr float kDefaultOutlierThresholdM = 0.014F;
inline constexpr int kDefaultOutlierNeighbors = 20;

// Immutable settings for GraspGenX preprocessing and gripper conditioning.
struct GraspInferenceOptions
{
  // Gripper names mapped to four XYZ triples: open extents, open offset, intermediate extents,
  // and intermediate offset, in meters. Uploaded with each request.
  std::unordered_map<std::string, std::array<float, 12>> grippers;

  // Positive, finite model scale factor applied to sampled XYZ after subtracting their mean.
  float kappa{kDefaultKappa};

  // Positive, finite spatial grid cell size, applied after kappa scaling when encoding points.
  float grid_size_m{kDefaultGridSizeM};

  // Positive, finite threshold in meters on mean L1 distance to neighbors, before kappa scaling.
  // Points whose mean distance reaches or exceeds the threshold are removed.
  float outlier_threshold_m{kDefaultOutlierThresholdM};

  // Requested nearest-neighbor count for outlier filtering, excluding self; must be in [1, 32].
  int outlier_neighbors{kDefaultOutlierNeighbors};
};

using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

// Model-specific CUDA preparation and decoding around an external tensor inference operation.
// The supplied operation publishes inputs to isaac_ros_tensor_rt and waits for its output.
// No TensorRT engine, execution context, or engine buffers are owned here.
class GraspInference final
{
public:
  using TensorInference = std::function<TensorList(const TensorList &)>;

  //! Validate preprocessing settings and retain the tensor transport callback.
  GraspInference(GraspInferenceOptions options, TensorInference inference);
  ~GraspInference();

  //! Prepare CUDA tensors, invoke the transport, and decode the matching outputs.
  //! Request-local allocations keep independent requests and late replies from aliasing buffers.
  //! Calls on this instance are serialized to reuse its CUDA stream safely.
  //! The caller must run this on a worker while the ROS executor services tensor callbacks.
  //! Throws on invalid inputs, transport failure, or CUDA errors.
  InferenceResult Infer(const InferenceRequest & request) const;

private:
  GraspInferenceOptions options_;
  TensorInference inference_;

  struct CudaStream;
  mutable std::mutex inference_mutex_;
  mutable std::unique_ptr<CudaStream> stream_;
};

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GRASPGENX__GRASP_INFERENCE_HPP_
