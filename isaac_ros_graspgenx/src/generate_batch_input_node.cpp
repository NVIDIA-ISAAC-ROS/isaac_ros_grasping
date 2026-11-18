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

#include "isaac_ros_graspgenx/generate_batch_input_node.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

GenerateBatchInputNode::GenerateBatchInputNode(const rclcpp::NodeOptions & options)
: Node("graspgenx_generate_batch_input", options)
{
  const auto input_topics = declare_parameter<std::vector<std::string>>(
    "input_topics", std::vector<std::string>{});
  if (input_topics.empty()) {
    throw std::invalid_argument("input_topics must name at least one object cloud topic");
  }
  latest_clouds_.resize(input_topics.size());
  batch_pub_ = create_publisher<CloudBatch>("/graspgenx/segmented_object_cloud_batch", 10);

  rclcpp::SubscriptionOptions cloud_options;
  cloud_options.acceptable_buffer_backends = "any";
  for (std::size_t index = 0; index < input_topics.size(); ++index) {
    cloud_subs_.push_back(
      create_subscription<PointCloud2>(
        input_topics[index], 10,
        [this, index](PointCloud2::ConstSharedPtr cloud) {
          auto batch = std::make_unique<CloudBatch>();
          {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_clouds_[index] = std::move(cloud);
            const auto & stamp = latest_clouds_[index]->header.stamp;
            for (const auto & latest : latest_clouds_) {
              if (!latest || latest->header.stamp != stamp) {
                return;
              }
            }
            for (const auto & latest : latest_clouds_) {
              batch->clouds.push_back(*latest);
            }
            std::fill(latest_clouds_.begin(), latest_clouds_.end(), nullptr);
          }
          batch_pub_->publish(std::move(batch));
        },
        cloud_options));
  }
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::graspgenx::GenerateBatchInputNode)
