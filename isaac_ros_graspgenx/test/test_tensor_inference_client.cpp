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

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "isaac_ros_graspgenx/generate_grasp_node.hpp"

namespace nvidia::isaac_ros::graspgenx
{
namespace
{
using namespace std::chrono_literals;
using Tensors = isaac_ros_tensor_msgs::msg::TensorList;

class TensorTransport : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    auto options = rclcpp::NodeOptions().use_intra_process_comms(true);
    node_ = std::make_shared<rclcpp::Node>("tensor_client_test", options);
    peer_ = std::make_shared<rclcpp::Node>("tensor_peer_test", options);
    output_ = peer_->create_publisher<Tensors>("graspgenx/inference/output", 10);
    input_ = peer_->create_subscription<Tensors>(
      "graspgenx/inference/input", 10,
      [this](Tensors::ConstSharedPtr input) {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        if (handler_) {handler_(*input);}
      });
    executor_.add_node(node_);
    executor_.add_node(peer_);
    spin_ = std::thread([this]() {executor_.spin();});
  }

  void TearDown() override
  {
    executor_.cancel();
    spin_.join();
    executor_.remove_node(node_);
    executor_.remove_node(peer_);
  }

  void Handle(std::function<void(const Tensors &)> handler)
  {
    std::lock_guard<std::mutex> lock(handler_mutex_);
    handler_ = std::move(handler);
  }

  rclcpp::Node::SharedPtr node_, peer_;
  rclcpp::Publisher<Tensors>::SharedPtr output_;
  rclcpp::Subscription<Tensors>::SharedPtr input_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_;
  std::mutex handler_mutex_;
  std::function<void(const Tensors &)> handler_;
};

TEST_F(TensorTransport, RoundTripUsesTopicAndPreservesInputHeader)
{
  TensorInferenceClient client(*node_, 2.0);
  Handle([this](const Tensors & request) {output_->publish(request);});
  Tensors inputs;
  inputs.header.frame_id = "object_frame";
  inputs.header.stamp.sec = 42;
  inputs.names = {"request_payload"};
  auto result = client.Infer(inputs);
  EXPECT_EQ(result.names, inputs.names);
  EXPECT_EQ(result.header.frame_id, inputs.header.frame_id);
  EXPECT_EQ(result.header.stamp, inputs.header.stamp);
  EXPECT_EQ(inputs.header.stamp.sec, 42);
}

TEST_F(TensorTransport, MismatchedLateReplyCannotCompleteNextRequest)
{
  TensorInferenceClient client(*node_, 0.3);
  std::promise<Tensors> first;
  auto first_request = first.get_future();
  Handle([&first](const Tensors & request) {first.set_value(request);});
  Tensors first_inputs;
  first_inputs.header.stamp.sec = 1;
  EXPECT_THROW(client.Infer(first_inputs), std::runtime_error);
  ASSERT_EQ(first_request.wait_for(1s), std::future_status::ready);
  auto stale = first_request.get();
  stale.names = {"stale"};
  Handle(
    [this, stale](const Tensors & request) {
      output_->publish(stale);
      auto current = request;
      current.names = {"current"};
      timer_ = peer_->create_wall_timer(
        30ms, [this, current]() {
          timer_->cancel();
          output_->publish(current);
        });
    });
  Tensors current_inputs;
  current_inputs.header.stamp.sec = 2;
  auto result = client.Infer(current_inputs);
  EXPECT_EQ(result.names, std::vector<std::string>{"current"});
  EXPECT_EQ(result.header.stamp, current_inputs.header.stamp);
}

TEST_F(TensorTransport, ConcurrentRequestIsRejectedWithoutLosingPendingReply)
{
  TensorInferenceClient client(*node_, 2.0);
  std::promise<Tensors> received;
  auto request = received.get_future();
  Handle([&received](const Tensors & input) {received.set_value(input);});
  auto pending = std::async(std::launch::async, [&client]() {return client.Infer(Tensors{});});
  ASSERT_EQ(request.wait_for(1s), std::future_status::ready);
  EXPECT_THROW(client.Infer(Tensors{}), std::runtime_error);
  output_->publish(request.get());
  ASSERT_EQ(pending.wait_for(1s), std::future_status::ready);
  EXPECT_NO_THROW(pending.get());
}

TEST_F(TensorTransport, MissingComponentTimesOutDuringDiscovery)
{
  input_.reset();
  output_.reset();
  TensorInferenceClient client(*node_, 0.05);
  EXPECT_THROW(client.Infer(Tensors{}), std::runtime_error);
  // A failed call must release the pending slot.
  EXPECT_THROW(client.Infer(Tensors{}), std::runtime_error);
}

TEST(TensorInferenceClientValidation, RejectsInvalidTimeout)
{
  auto context = std::make_shared<rclcpp::Context>();
  context->init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>(
    "tensor_client_validation_test",
    rclcpp::NodeOptions().context(context));

  EXPECT_THROW(TensorInferenceClient(*node, 0.0), std::invalid_argument);

  context->shutdown("validation test complete");
}

}  // namespace
}  // namespace nvidia::isaac_ros::graspgenx
