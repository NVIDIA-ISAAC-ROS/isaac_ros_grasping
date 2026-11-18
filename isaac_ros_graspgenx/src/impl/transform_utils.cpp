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

#include "isaac_ros_graspgenx/impl/transform_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

constexpr double kMinQuaternionNorm = 1.0e-12;

}  // namespace

Eigen::Isometry3d PoseToIsometry(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Quaterniond rotation(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  const Eigen::Vector3d translation(
    pose.position.x, pose.position.y, pose.position.z);
  if (!rotation.coeffs().allFinite() || !translation.allFinite()) {
    throw std::invalid_argument("Pose contains a nonfinite value");
  }
  if (rotation.norm() <= kMinQuaternionNorm) {
    throw std::invalid_argument("Pose quaternion has zero norm");
  }
  // Normalize the quaternion so the resulting linear component is a rigid rotation.
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = rotation.normalized().toRotationMatrix();
  transform.translation() = translation;
  return transform;
}

geometry_msgs::msg::Pose IsometryToPose(const Eigen::Isometry3d & transform)
{
  // A geometry pose must not encode scale, reflection, shear, or nonfinite values.
  if (!IsFiniteRigidTransform(transform)) {
    throw std::invalid_argument("Cannot convert a nonfinite or non-rigid transform to Pose");
  }
  const Eigen::Quaterniond rotation(transform.linear());
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  pose.orientation.x = rotation.x();
  pose.orientation.y = rotation.y();
  pose.orientation.z = rotation.z();
  pose.orientation.w = rotation.w();
  return pose;
}

Eigen::Isometry3d TransformToIsometry(const geometry_msgs::msg::Transform & transform_msg)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform_msg.translation.x;
  pose.position.y = transform_msg.translation.y;
  pose.position.z = transform_msg.translation.z;
  pose.orientation = transform_msg.rotation;
  return PoseToIsometry(pose);
}

bool IsFiniteRigidTransform(
  const Eigen::Isometry3d & transform,
  const double tolerance)
{
  if (!std::isfinite(tolerance) || tolerance < 0.0 || !transform.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = transform.linear();
  // A proper rotation is orthonormal and has determinant +1 within tolerance.
  return
    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <= tolerance &&
    std::abs(rotation.determinant() - 1.0) <= tolerance;
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia
