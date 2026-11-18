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

#include <cmath>
#include <stdexcept>
#include <utility>

#include "isaac_ros_graspgenx/generate_grasp_node.hpp"

namespace nvidia::isaac_ros::graspgenx
{

TensorInferenceClient::TensorInferenceClient(rclcpp::Node & node, const double timeout_sec)
: state_(std::make_shared<State>()), context_(node.get_node_base_interface()->get_context())
{
  if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0 || timeout_sec > 3600.0) {
    throw std::invalid_argument("inference_timeout_sec must be in (0, 3600]");
  }
  timeout_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(timeout_sec));
  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  publisher_ = node.create_publisher<TensorList>(
    "graspgenx/inference/input", rclcpp::QoS(1).reliable(), publisher_options);
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  subscription_ = node.create_subscription<TensorList>(
    "graspgenx/inference/output", rclcpp::QoS(1).reliable(),
    [state = state_](TensorList::ConstSharedPtr output) {
      std::lock_guard<std::mutex> lock(state->mutex);
      // TensorRTNode preserves the input header; reject outputs for a different cloud.
      if (state->pending && output->header == state->header && !state->output) {
        state->output = std::move(output);
        state->ready.notify_all();
      }
    }, subscription_options);
}

TensorInferenceClient::TensorList TensorInferenceClient::Infer(const TensorList & inputs)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout_;
  std::unique_lock<std::mutex> lock(state_->mutex);
  if (state_->pending) {
    throw std::runtime_error("A TensorRT request is already pending");
  }
  auto message = std::make_unique<TensorList>(inputs);
  state_->header = message->header;
  state_->output.reset();
  state_->pending = true;
  const auto wait = [&]() {
      if (!rclcpp::ok(context_)) {
        throw std::runtime_error("ROS shut down while waiting for TensorRT inference");
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
                "TensorRT inference timed out; check the tensor_rt component "
                "and its engine/plugin");
      }
      state_->ready.wait_for(lock, std::chrono::milliseconds(10));
    };
  try {
    // Reliable volatile topics do not retain requests published before discovery.
    while (publisher_->get_subscription_count() == 0 ||
      subscription_->get_publisher_count() == 0)
    {
      wait();
    }
    lock.unlock();
    publisher_->publish(std::move(message));
    lock.lock();
    while (!state_->output) {wait();}
    TensorList result = *state_->output;
    state_->output.reset();
    state_->pending = false;
    return result;
  } catch (...) {
    if (!lock.owns_lock()) {lock.lock();}
    state_->pending = false;
    state_->output.reset();
    throw;
  }
}

}  // namespace nvidia::isaac_ros::graspgenx
