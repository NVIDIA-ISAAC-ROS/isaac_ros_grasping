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

#ifndef ISAAC_ROS_GRASPGENX__GENERATE_BATCH_INPUT_NODE_HPP_
#define ISAAC_ROS_GRASPGENX__GENERATE_BATCH_INPUT_NODE_HPP_

#include <mutex>
#include <vector>

#include "isaac_ros_graspgenx_interfaces/msg/segmented_object_cloud_batch.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

// Publishes the batch input for GenerateGraspNode once every object cloud topic has a cloud with
// the same stamp. Compose it with GenerateGraspNode using intra-process communication: clouds
// nested in a batch keep their CUDA buffers only on the intra-process path.
class GenerateBatchInputNode : public rclcpp::Node
{
public:
  explicit GenerateBatchInputNode(const rclcpp::NodeOptions & options);

private:
  using CloudBatch = isaac_ros_graspgenx_interfaces::msg::SegmentedObjectCloudBatch;
  using PointCloud2 = sensor_msgs::msg::PointCloud2;

  std::mutex mutex_;
  std::vector<PointCloud2::ConstSharedPtr> latest_clouds_;
  std::vector<rclcpp::Subscription<PointCloud2>::SharedPtr> cloud_subs_;
  rclcpp::Publisher<CloudBatch>::SharedPtr batch_pub_;
};

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GRASPGENX__GENERATE_BATCH_INPUT_NODE_HPP_
