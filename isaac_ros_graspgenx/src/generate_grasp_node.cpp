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

#include "isaac_ros_graspgenx/generate_grasp_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "isaac_ros_graspgenx/grasp_inference.hpp"
#include "isaac_ros_graspgenx/impl/transform_utils.hpp"
#include "isaac_ros_graspgenx_interfaces/msg/grasp_candidate.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

const char * GraspSourceToString(GraspSource source)
{
  switch (source) {
    case GraspSource::kDiffusion:
      return "diffusion";
  }
  throw std::invalid_argument("Unknown grasp source");
}

class BusyReset
{
public:
  explicit BusyReset(std::atomic_bool & busy)
  : busy_(busy) {}

  ~BusyReset() {busy_.store(false);}

  BusyReset(const BusyReset &) = delete;
  BusyReset & operator=(const BusyReset &) = delete;

private:
  std::atomic_bool & busy_;
};

template<typename ActionT>
void PublishFeedback(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionT>> & goal_handle,
  const std::string & status,
  const float progress)
{
  auto feedback = std::make_shared<typename ActionT::Feedback>();
  feedback->status = status;
  feedback->progress = progress;
  goal_handle->publish_feedback(feedback);
}

}  // namespace

GenerateGraspNode::GenerateGraspNode(const rclcpp::NodeOptions & options)
: Node("graspgenx_generate_grasp_node", options),
  origin_frame_(declare_parameter<std::string>("origin_frame", "base_link")),
  default_gripper_(declare_parameter<std::string>("default_gripper_name", "robotiq_2f_140")),
  tf_timeout_sec_(declare_parameter<double>("tf_timeout_sec", 2.0)),
  min_object_points_(static_cast<std::size_t>(
      std::max<int64_t>(1, declare_parameter<int64_t>("min_object_points", 100))))
{
  if (origin_frame_.empty()) {
    throw std::invalid_argument("origin_frame must not be empty");
  }
  if (default_gripper_.empty()) {
    throw std::invalid_argument("default_gripper_name must not be empty");
  }
  if (!std::isfinite(tf_timeout_sec_) || tf_timeout_sec_ < 0.0) {
    throw std::invalid_argument("tf_timeout_sec must be finite and nonnegative");
  }

  // Fixed transform from the predicted grasp frame to the tool frame:
  // [x, y, z] in meters followed by quaternion [qx, qy, qz, qw].
  const auto grasp_pose_tool = declare_parameter<std::vector<double>>(
    "grasp_pose_tool",
    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  if (grasp_pose_tool.size() != 7U) {
    throw std::invalid_argument(
            "grasp_pose_tool must contain [x,y,z,qx,qy,qz,qw]");
  }
  grasp_pose_tool_.position.x = grasp_pose_tool[0];
  grasp_pose_tool_.position.y = grasp_pose_tool[1];
  grasp_pose_tool_.position.z = grasp_pose_tool[2];
  grasp_pose_tool_.orientation.x = grasp_pose_tool[3];
  grasp_pose_tool_.orientation.y = grasp_pose_tool[4];
  grasp_pose_tool_.orientation.z = grasp_pose_tool[5];
  grasp_pose_tool_.orientation.w = grasp_pose_tool[6];

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  GraspInferenceOptions backend_options;
  backend_options.kappa =
    static_cast<float>(declare_parameter<double>("kappa", kDefaultKappa));
  backend_options.grid_size_m =
    static_cast<float>(declare_parameter<double>("grid_size_m", kDefaultGridSizeM));
  // Model conditioning describes the gripper sweep as four XYZ triples in meters:
  // open extents, open offset, intermediate extents, and intermediate offset.
  // These defaults describe the robotiq_2f_140 gripper. Configure this parameter for other
  // grippers.
  const auto sweep = declare_parameter<std::vector<double>>(
    "gripper_sweep_volume", std::vector<double>{
          0.125, 0.0320000015, 0.061999999, 0.0, 0.0, 0.174999997,
          0.0659999996, 0.0320000015, 0.061999999, 0.0, 0.0, 0.194999993});
  if (sweep.size() != 12U ||
    !std::all_of(sweep.begin(), sweep.end(), [](double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("gripper_sweep_volume must contain 12 finite values");
  }
  std::array<float, 12> gripper{};
  std::copy(sweep.begin(), sweep.end(), gripper.begin());
  backend_options.grippers.emplace(default_gripper_, gripper);
  tensor_client_ = std::make_unique<TensorInferenceClient>(
    *this, declare_parameter<double>("inference_timeout_sec", 60.0));
  inference_backend_ = std::make_unique<GraspInference>(
    std::move(backend_options),
    [this](const TensorList & inputs) {return tensor_client_->Infer(inputs);});
  candidate_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
    "/graspgenx/grasp_candidates",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

  rclcpp::SubscriptionOptions cloud_options;
  cloud_options.acceptable_buffer_backends = "any";
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/graspgenx/segmented_object_cloud", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr incoming_cloud) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      cloud_ = std::move(incoming_cloud);
    },
    cloud_options);
  cloud_batch_sub_ = create_subscription<CloudBatch>(
    "/graspgenx/segmented_object_cloud_batch", rclcpp::SensorDataQoS(),
    [this](CloudBatch::ConstSharedPtr incoming_batch) {
      if (incoming_batch->clouds.empty()) {
        RCLCPP_WARN(get_logger(), "Dropping empty segmented object cloud batch");
        return;
      }
      std::lock_guard<std::mutex> lock(cloud_batch_mutex_);
      cloud_batch_ = std::move(incoming_batch);
    },
    cloud_options);

  batch_action_server_ = rclcpp_action::create_server<GenerateGraspsBatch>(
    this, "/graspgenx/generate_grasps_batch",
    std::bind(
      &GenerateGraspNode::HandleBatchGoal, this,
      std::placeholders::_1, std::placeholders::_2),
    [](const std::shared_ptr<BatchGoalHandle>) {
      return rclcpp_action::CancelResponse::REJECT;
    },
    std::bind(
      &GenerateGraspNode::HandleBatchAccepted, this, std::placeholders::_1));

  // Actions carry only cloud metadata; DDS action transport would download the payload.
  action_server_ = rclcpp_action::create_server<GenerateGrasps>(
    this, "/graspgenx/generate_grasps",
    std::bind(
      &GenerateGraspNode::HandleGoal, this,
      std::placeholders::_1, std::placeholders::_2),
    [](const std::shared_ptr<GoalHandle>) {
      return rclcpp_action::CancelResponse::REJECT;
    },
    std::bind(
      &GenerateGraspNode::HandleAccepted, this, std::placeholders::_1));
}

rclcpp_action::GoalResponse GenerateGraspNode::HandleGoal(
  const rclcpp_action::GoalUUID &,
  std::shared_ptr<const GenerateGrasps::Goal> goal)
{
  if (goal->object.object_id == 0 || goal->num_grasps <= 0 || goal->top_k_grasps < -1 ||
    !std::isfinite(goal->grasp_confidence_threshold))
  {
    RCLCPP_WARN(get_logger(), "Rejecting invalid grasp goal");
    return rclcpp_action::GoalResponse::REJECT;
  }
  // Claim the single-flight backend slot during goal admission.
  bool expected = false;
  if (!action_server_busy_.compare_exchange_strong(expected, true)) {
    RCLCPP_WARN(get_logger(), "Rejecting grasp goal: server is busy");
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::GoalResponse GenerateGraspNode::HandleBatchGoal(
  const rclcpp_action::GoalUUID &,
  std::shared_ptr<const GenerateGraspsBatch::Goal> goal)
{
  if (goal->num_grasps <= 0 || goal->top_k_grasps < -1 ||
    !std::isfinite(goal->grasp_confidence_threshold))
  {
    RCLCPP_WARN(get_logger(), "Rejecting invalid batch grasp goal");
    return rclcpp_action::GoalResponse::REJECT;
  }

  bool expected = false;
  if (!action_server_busy_.compare_exchange_strong(expected, true)) {
    RCLCPP_WARN(get_logger(), "Rejecting batch grasp goal: server is busy");
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

void GenerateGraspNode::HandleBatchAccepted(
  const std::shared_ptr<BatchGoalHandle> goal_handle)
{
  const auto self = std::static_pointer_cast<GenerateGraspNode>(shared_from_this());
  std::thread([self, goal_handle]() {self->ExecuteBatch(goal_handle);}).detach();
}

void GenerateGraspNode::HandleAccepted(
  const std::shared_ptr<GoalHandle> goal_handle)
{
  // Keep the node and its backend alive until non-preemptible inference finishes.
  const auto self = std::static_pointer_cast<GenerateGraspNode>(shared_from_this());
  std::thread([self, goal_handle]() {self->Execute(goal_handle);}).detach();
}

void GenerateGraspNode::Execute(
  const std::shared_ptr<GoalHandle> goal_handle)
{
  // Hold the single-flight slot through completion, including non-preemptible backend work.
  BusyReset reset(action_server_busy_);
  const auto result = std::make_shared<GenerateGrasps::Result>();
  const auto start = std::chrono::steady_clock::now();

  PublishFeedback<GenerateGrasps>(goal_handle, "validating_input", 0.1F);
  const auto & goal = goal_handle->get_goal();
  PublishFeedback<GenerateGrasps>(goal_handle, "running_inference", 0.5F);
  result->object_result = ProcessCloud(
    goal->object.object_id,
    GetSegmentedObjectPointCloud(),
    goal->gripper_name.empty() ? default_gripper_ : goal->gripper_name,
    goal->grasp_confidence_threshold, goal->num_grasps, goal->top_k_grasps,
    goal->random_seed, result->message);

  PublishFeedback<GenerateGrasps>(goal_handle, "ranking_candidates", 0.9F);
  if (result->object_result.success) {
    PublishCandidates(result->object_result);
    result->processing_time_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    goal_handle->succeed(result);
  } else {
    result->processing_time_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    goal_handle->abort(result);
  }
}

void GenerateGraspNode::ExecuteBatch(
  const std::shared_ptr<BatchGoalHandle> goal_handle)
{
  BusyReset reset(action_server_busy_);
  const auto action_result = std::make_shared<GenerateGraspsBatch::Result>();
  const auto & goal = goal_handle->get_goal();
  const auto start = std::chrono::steady_clock::now();

  CloudBatch::ConstSharedPtr batch;
  {
    std::lock_guard<std::mutex> lock(cloud_batch_mutex_);
    batch = cloud_batch_;
  }
  if (!batch) {
    action_result->message = "No segmented object cloud batch is available";
    action_result->processing_time_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    goal_handle->abort(action_result);
    return;
  }

  auto & results = action_result->object_results;
  results.resize(batch->clouds.size());
  std::size_t completed = 0U;
  std::string details;
  for (std::size_t index = 0; index < batch->clouds.size(); ++index) {
    const uint64_t object_seed_bits = static_cast<uint64_t>(goal->random_seed) + index;
    const int64_t object_seed = static_cast<int64_t>(
      object_seed_bits & static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    std::string message;
    results[index] = ProcessCloud(
      index + 1U,
      sensor_msgs::msg::PointCloud2::ConstSharedPtr(batch, &batch->clouds[index]),
      goal->gripper_name.empty() ? default_gripper_ : goal->gripper_name,
      goal->grasp_confidence_threshold, goal->num_grasps, goal->top_k_grasps,
      object_seed, message);
    const auto & processed = results[index];
    if (!processed.success) {
      details += "; object " + std::to_string(index + 1U) + ": " + message;
    }
    ++completed;
  }

  const std::size_t succeeded = static_cast<std::size_t>(std::count_if(
      results.begin(), results.end(), [](const auto & result) {return result.success;}));
  action_result->all_objects_succeeded = succeeded == results.size();
  action_result->processing_time_sec =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  action_result->message =
    "Processed " + std::to_string(completed) + " of " + std::to_string(results.size()) +
    " objects; " + std::to_string(succeeded) + " succeeded" + details;
  goal_handle->succeed(action_result);
}

sensor_msgs::msg::PointCloud2::ConstSharedPtr GenerateGraspNode::GetSegmentedObjectPointCloud()
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);
  return cloud_;
}

void GenerateGraspNode::PublishCandidates(const ObjectGraspResult & result)
{
  geometry_msgs::msg::PoseArray message;
  message.header = result.candidates.front().pose.header;
  message.poses.reserve(result.candidates.size());
  for (const auto & candidate : result.candidates) {
    message.poses.push_back(candidate.pose.pose);
  }
  candidate_pub_->publish(message);
}

GenerateGraspNode::ObjectGraspResult GenerateGraspNode::ProcessCloud(
  const uint64_t segmented_object_point_cloud_id,
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud,
  const std::string & gripper_name,
  const float threshold,
  const int32_t num_grasps,
  const int32_t top_k_grasps,
  const int64_t random_seed,
  std::string & message)
{
  ObjectGraspResult output;
  output.object_id = segmented_object_point_cloud_id;
  std_msgs::msg::Header output_header;
  output_header.frame_id = origin_frame_;
  output_header.stamp = now();
  try {
    if (!cloud) {
      throw std::invalid_argument("No segmented object cloud is available");
    }
    if (cloud->header.frame_id.empty()) {
      throw std::invalid_argument("Input PointCloud2 frame_id is empty");
    }
    const auto count = static_cast<std::size_t>(cloud->width) * cloud->height;
    if (count < min_object_points_) {
      throw std::invalid_argument("Insufficient object points: " + std::to_string(count));
    }
    InferenceRequest request;
    request.object_cloud = std::move(cloud);
    request.gripper_name = gripper_name;
    request.grasp_confidence_threshold = threshold;
    request.num_grasps = num_grasps;
    request.top_k_grasps = top_k_grasps;
    request.random_seed = random_seed;
    request.grasp_pose_tool = PoseToIsometry(grasp_pose_tool_);
    const auto & header = request.object_cloud->header;
    if (header.frame_id != origin_frame_) {
      try {
        const auto transform = tf_buffer_->lookupTransform(
          origin_frame_, header.frame_id, rclcpp::Time(header.stamp),
          rclcpp::Duration::from_seconds(tf_timeout_sec_));
        request.origin_pose_point_cloud = TransformToIsometry(transform.transform);
      } catch (const tf2::TransformException & error) {
        RCLCPP_ERROR(
          get_logger(), "Failed to transform object cloud from '%s' to '%s': %s",
          header.frame_id.c_str(), origin_frame_.c_str(), error.what());
        throw;
      }
    }
    // Only TF metadata is prepared here; the ROS payload is consumed on the GPU.
    const auto inference = inference_backend_->Infer(request);
    output.success = inference.success;
    message = inference.message;
    if (inference.success) {
      output.candidates.reserve(inference.grasps.size());
      for (const auto & grasp : inference.grasps) {
        isaac_ros_graspgenx_interfaces::msg::GraspCandidate candidate;
        candidate.pose.header = output_header;
        candidate.pose.pose = IsometryToPose(grasp.grasp_pose);
        candidate.confidence = grasp.confidence;
        candidate.source = GraspSourceToString(grasp.source);
        output.candidates.push_back(std::move(candidate));
      }
    }
  } catch (const std::exception & error) {
    message = error.what();
  }
  return output;
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::graspgenx::GenerateGraspNode)
