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

#ifndef ISAAC_ROS_GRASPGENX__GENERATE_GRASP_NODE_HPP_
#define ISAAC_ROS_GRASPGENX__GENERATE_GRASP_NODE_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "isaac_ros_graspgenx_interfaces/action/generate_grasps.hpp"
#include "isaac_ros_graspgenx_interfaces/action/generate_grasps_batch.hpp"
#include "isaac_ros_graspgenx_interfaces/msg/object_grasp_result.hpp"
#include "isaac_ros_graspgenx_interfaces/msg/segmented_object_cloud_batch.hpp"
#include "isaac_ros_graspgenx/grasp_inference.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{

// Request/reply adapter for the standard TensorRTNode tensor topics. The action worker waits
// here while the executor services callbacks. The public action/cloud timestamps are unaffected.
class TensorInferenceClient
{
public:
  using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

  TensorInferenceClient(rclcpp::Node & node, const double timeout_sec);

  //! Publish one request and wait for its matching output, including discovery in the deadline.
  //! Throws on timeout, shutdown, or concurrent use. A timed-out request may still run in TRT;
  //! its message retains the GPU allocations until consumed. A later request that reuses the same
  //! input header cannot be distinguished from that timed-out request's late reply.
  TensorList Infer(const TensorList & inputs);

private:
  struct State
  {
    std::mutex mutex;
    std::condition_variable ready;
    bool pending{false};
    std_msgs::msg::Header header;
    TensorList::ConstSharedPtr output;
  };

  std::shared_ptr<State> state_;
  rclcpp::Context::SharedPtr context_;
  std::chrono::steady_clock::duration timeout_;
  rclcpp::Publisher<TensorList>::SharedPtr publisher_;
  rclcpp::Subscription<TensorList>::SharedPtr subscription_;
};

// Serves actions for the latest single-cloud and batch inputs.
class GenerateGraspNode : public rclcpp::Node
{
public:
  explicit GenerateGraspNode(const rclcpp::NodeOptions & options);

private:
  // ROS interface types owned by this node.
  using GenerateGrasps = isaac_ros_graspgenx_interfaces::action::GenerateGrasps;
  using GenerateGraspsBatch = isaac_ros_graspgenx_interfaces::action::GenerateGraspsBatch;
  using ObjectGraspResult = isaac_ros_graspgenx_interfaces::msg::ObjectGraspResult;
  using CloudBatch = isaac_ros_graspgenx_interfaces::msg::SegmentedObjectCloudBatch;
  using BatchGoalHandle = rclcpp_action::ServerGoalHandle<GenerateGraspsBatch>;
  using GoalHandle = rclcpp_action::ServerGoalHandle<GenerateGrasps>;

  // Action callbacks validate admission and start asynchronous execution.
  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const GenerateGrasps::Goal> goal);
  void HandleAccepted(const std::shared_ptr<GoalHandle> goal_handle);
  void Execute(const std::shared_ptr<GoalHandle> goal_handle);

  rclcpp_action::GoalResponse HandleBatchGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const GenerateGraspsBatch::Goal> goal);
  void HandleBatchAccepted(const std::shared_ptr<BatchGoalHandle> goal_handle);
  void ExecuteBatch(const std::shared_ptr<BatchGoalHandle> goal_handle);

  sensor_msgs::msg::PointCloud2::ConstSharedPtr GetSegmentedObjectPointCloud();

  void PublishCandidates(const ObjectGraspResult & result);

  // Processes one object cloud; writes diagnostics to the action-level message.
  ObjectGraspResult ProcessCloud(
    uint64_t segmented_object_point_cloud_id,
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud,
    const std::string & gripper_name,
    float threshold,
    int32_t num_grasps,
    int32_t top_k_grasps,
    int64_t random_seed,
    std::string & message);

  // Request policy and transforms; the TF timeout is in seconds.
  std::string origin_frame_;
  std::string default_gripper_;
  double tf_timeout_sec_;
  std::size_t min_object_points_;
  geometry_msgs::msg::Pose grasp_pose_tool_;
  std::atomic_bool action_server_busy_{false};

  std::mutex cloud_mutex_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_;
  std::mutex cloud_batch_mutex_;
  CloudBatch::ConstSharedPtr cloud_batch_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<CloudBatch>::SharedPtr cloud_batch_sub_;

  // ROS and TF resources are owned for the lifetime of the node.
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<TensorInferenceClient> tensor_client_;
  std::unique_ptr<GraspInference> inference_backend_;
  rclcpp_action::Server<GenerateGrasps>::SharedPtr action_server_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr candidate_pub_;
  rclcpp_action::Server<GenerateGraspsBatch>::SharedPtr batch_action_server_;
};

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GRASPGENX__GENERATE_GRASP_NODE_HPP_
