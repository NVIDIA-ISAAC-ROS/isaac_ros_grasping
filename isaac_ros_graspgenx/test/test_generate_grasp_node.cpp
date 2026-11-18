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

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "isaac_ros_graspgenx/generate_grasp_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

using namespace std::chrono_literals;

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

rclcpp::NodeOptions OptionsWith(
  const std::vector<rclcpp::Parameter> & overrides)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(overrides);
  return options;
}

class ExecutorSpinGuard
{
public:
  explicit ExecutorSpinGuard(rclcpp::Executor & executor)
  : executor_(executor), spin_thread_([this]() {executor_.spin();})
  {
  }

  ~ExecutorSpinGuard()
  {
    executor_.cancel();
    spin_thread_.join();
  }

  ExecutorSpinGuard(const ExecutorSpinGuard &) = delete;
  ExecutorSpinGuard & operator=(const ExecutorSpinGuard &) = delete;

private:
  rclcpp::Executor & executor_;
  std::thread spin_thread_;
};

class GenerateGraspsConstructorTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

TEST_F(GenerateGraspsConstructorTest, RejectsInvalidConfiguration)
{
  EXPECT_THROW(
    GenerateGraspNode(
      OptionsWith({rclcpp::Parameter("origin_frame", "")})),
    std::invalid_argument);
  EXPECT_THROW(
    GenerateGraspNode(
      OptionsWith({rclcpp::Parameter("default_gripper_name", "")})),
    std::invalid_argument);
  EXPECT_THROW(
    GenerateGraspNode(
      OptionsWith(
        {rclcpp::Parameter(
            "grasp_pose_tool",
            std::vector<double>{0.0, 0.0, 0.0})})),
    std::invalid_argument);
}

TEST_F(GenerateGraspsConstructorTest, ActionWithoutCloudAborts)
{
  using Action = isaac_ros_graspgenx_interfaces::action::GenerateGrasps;

  auto io_node = std::make_shared<rclcpp::Node>("graspgenx_no_cloud_test");
  auto action_client =
    rclcpp_action::create_client<Action>(io_node, "/graspgenx/generate_grasps");
  auto server = std::make_shared<GenerateGraspNode>(rclcpp::NodeOptions());

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(io_node);
  executor.add_node(server);
  ExecutorSpinGuard spin_guard(executor);

  Action::Goal goal;
  goal.object.object_id = 1U;
  goal.num_grasps = 1;
  goal.top_k_grasps = -1;
  goal.grasp_confidence_threshold = -1.0F;
  ASSERT_TRUE(action_client->wait_for_action_server(2s));
  auto sent = action_client->async_send_goal(goal);
  ASSERT_EQ(sent.wait_for(2s), std::future_status::ready);
  const auto goal_handle = sent.get();
  ASSERT_NE(goal_handle, nullptr);
  auto pending = action_client->async_get_result(goal_handle);
  ASSERT_EQ(pending.wait_for(2s), std::future_status::ready);
  const auto result = pending.get();
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->message, "No segmented object cloud is available");
}

TEST_F(GenerateGraspsConstructorTest, BatchUsesLatestCloudBatch)
{
  constexpr std::size_t kObjectCount = 9U;
  using BatchAction = isaac_ros_graspgenx_interfaces::action::GenerateGraspsBatch;
  using CloudBatch = isaac_ros_graspgenx_interfaces::msg::SegmentedObjectCloudBatch;

  auto io_node = std::make_shared<rclcpp::Node>("graspgenx_latest_group_test");
  auto cloud_batch_pub = io_node->create_publisher<CloudBatch>(
    "/graspgenx/segmented_object_cloud_batch", rclcpp::SensorDataQoS());
  auto batch_client =
    rclcpp_action::create_client<BatchAction>(io_node, "/graspgenx/generate_grasps_batch");
  auto server = std::make_shared<GenerateGraspNode>(rclcpp::NodeOptions());

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(io_node);
  executor.add_node(server);

  const auto discovery_deadline = std::chrono::steady_clock::now() + 2s;
  while (cloud_batch_pub->get_subscription_count() == 0U &&
    std::chrono::steady_clock::now() < discovery_deadline)
  {
    std::this_thread::yield();
  }
  ASSERT_GT(cloud_batch_pub->get_subscription_count(), 0U);

  CloudBatch batch;
  batch.clouds.resize(kObjectCount);
  for (std::size_t index = 0; index < batch.clouds.size(); ++index) {
    batch.clouds[index].header.stamp.sec = static_cast<int32_t>(index + 1U);
    batch.clouds[index].header.frame_id = "object_" + std::to_string(index);
  }
  cloud_batch_pub->publish(batch);
  executor.spin_once(2s);

  BatchAction::Goal goal;
  goal.num_grasps = 1;
  goal.top_k_grasps = -1;
  goal.grasp_confidence_threshold = -1.0F;
  ExecutorSpinGuard spin_guard(executor);
  ASSERT_TRUE(batch_client->wait_for_action_server(2s));
  auto sent = batch_client->async_send_goal(goal);
  ASSERT_EQ(sent.wait_for(2s), std::future_status::ready);
  const auto goal_handle = sent.get();
  ASSERT_NE(goal_handle, nullptr);
  auto pending = batch_client->async_get_result(goal_handle);
  ASSERT_EQ(pending.wait_for(2s), std::future_status::ready);
  const auto result = pending.get();
  ASSERT_NE(result.result, nullptr);
  ASSERT_EQ(result.result->object_results.size(), kObjectCount);
  EXPECT_TRUE(std::none_of(
      result.result->object_results.begin(), result.result->object_results.end(),
      [](const auto & object_result) {return object_result.success;}));
  EXPECT_NE(result.result->message.find("Insufficient object points"), std::string::npos);
  EXPECT_EQ(result.result->message.find("not in the latest"), std::string::npos);
}

}  // namespace
}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia
