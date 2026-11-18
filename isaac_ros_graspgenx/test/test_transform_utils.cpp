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

#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"
#include "isaac_ros_graspgenx/impl/transform_utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

constexpr double kHalfPi = 1.57079632679489661923;

}  // namespace

TEST(TransformUtils, ConvertsPoseAndNormalizesQuaternion)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = 1.0;
  pose.position.y = -2.0;
  pose.position.z = 3.0;
  pose.orientation.z = std::sqrt(2.0);
  pose.orientation.w = std::sqrt(2.0);

  const auto transform = PoseToIsometry(pose);
  const auto round_trip = IsometryToPose(transform);

  EXPECT_TRUE(transform.translation().isApprox(Eigen::Vector3d(1.0, -2.0, 3.0)));
  EXPECT_TRUE(
    transform.linear().isApprox(
      Eigen::AngleAxisd(kHalfPi, Eigen::Vector3d::UnitZ()).toRotationMatrix()));
  EXPECT_NEAR(round_trip.orientation.z, std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(round_trip.orientation.w, std::sqrt(0.5), 1.0e-12);
}

TEST(TransformUtils, ConvertsTransformMessage)
{
  geometry_msgs::msg::Transform message;
  message.translation.x = 0.5;
  message.translation.y = 1.0;
  message.translation.z = -0.5;
  message.rotation.w = 1.0;

  const auto transform = TransformToIsometry(message);

  EXPECT_TRUE(transform.linear().isIdentity());
  EXPECT_TRUE(transform.translation().isApprox(Eigen::Vector3d(0.5, 1.0, -0.5)));
}

TEST(TransformUtils, IdentifiesInvalidRigidTransforms)
{
  EXPECT_TRUE(IsFiniteRigidTransform(Eigen::Isometry3d::Identity()));

  Eigen::Isometry3d scaled = Eigen::Isometry3d::Identity();
  scaled.linear()(0, 0) = 2.0;
  EXPECT_FALSE(IsFiniteRigidTransform(scaled));
  EXPECT_THROW(IsometryToPose(scaled), std::invalid_argument);

  Eigen::Isometry3d reflected = Eigen::Isometry3d::Identity();
  reflected.linear()(0, 0) = -1.0;
  EXPECT_FALSE(IsFiniteRigidTransform(reflected));
  EXPECT_FALSE(IsFiniteRigidTransform(Eigen::Isometry3d::Identity(), -1.0));
}

TEST(TransformUtils, RejectsNonfiniteAndZeroNormInputs)
{
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 0.0;
  EXPECT_THROW(PoseToIsometry(pose), std::invalid_argument);
  pose.orientation.w = 1.0;
  pose.position.x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(PoseToIsometry(pose), std::invalid_argument);
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia
