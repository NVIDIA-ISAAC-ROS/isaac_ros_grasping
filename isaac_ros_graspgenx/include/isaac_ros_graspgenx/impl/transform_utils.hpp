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

#ifndef ISAAC_ROS_GRASPGENX__IMPL__TRANSFORM_UTILS_HPP_
#define ISAAC_ROS_GRASPGENX__IMPL__TRANSFORM_UTILS_HPP_

#include <Eigen/Geometry>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

// Converts between ROS pose messages and double-precision rigid transforms.
Eigen::Isometry3d PoseToIsometry(const geometry_msgs::msg::Pose & pose);
geometry_msgs::msg::Pose IsometryToPose(const Eigen::Isometry3d & transform);
Eigen::Isometry3d TransformToIsometry(const geometry_msgs::msg::Transform & transform);

// Checks finite coefficients and rigid-rotation invariants within tolerance.
bool IsFiniteRigidTransform(
  const Eigen::Isometry3d & transform,
  const double tolerance = 1.0e-3);

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GRASPGENX__IMPL__TRANSFORM_UTILS_HPP_
