# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Send one GraspGen-X action goal for the latest complete cloud batch."""

from isaac_ros_graspgenx_interfaces.action import GenerateGraspsBatch
from isaac_ros_graspgenx_interfaces.msg import SegmentedObjectCloudBatch
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class GraspGenXBatchClient(Node):
    """Submit one inference request after a complete cloud batch arrives."""

    def __init__(self) -> None:
        super().__init__('graspgenx_batch_client')
        self.declare_parameter('gripper_name', 'robotiq_2f_140')
        self.declare_parameter('num_grasps', 50)
        self.declare_parameter('grasp_confidence_threshold', -1.0)
        self.declare_parameter('top_k_grasps', 20)
        self.declare_parameter('random_seed', 0)

        self._batch_size = 0
        self._batch_ready = False
        self._goal_sent = False
        self._action_client = ActionClient(
            self, GenerateGraspsBatch, '/graspgenx/generate_grasps_batch')
        self._batch_sub = self.create_subscription(
            SegmentedObjectCloudBatch,
            '/graspgenx/segmented_object_cloud_batch',
            self._batch_callback,
            qos_profile_sensor_data,
        )
        self._send_timer = self.create_timer(0.5, self._try_send_goal)

    def _batch_callback(self, batch: SegmentedObjectCloudBatch) -> None:
        if self._goal_sent or not batch.clouds:
            return
        self._batch_size = len(batch.clouds)
        self._batch_ready = True
        self.get_logger().info(f'Received batch with {self._batch_size} objects')
        self._try_send_goal()

    def _try_send_goal(self) -> None:
        if self._goal_sent or not self._batch_ready:
            return
        if not self._action_client.server_is_ready():
            return

        goal = GenerateGraspsBatch.Goal()
        goal.gripper_name = str(self.get_parameter('gripper_name').value)
        goal.num_grasps = int(self.get_parameter('num_grasps').value)
        goal.grasp_confidence_threshold = float(
            self.get_parameter('grasp_confidence_threshold').value)
        goal.top_k_grasps = int(self.get_parameter('top_k_grasps').value)
        goal.random_seed = int(self.get_parameter('random_seed').value)

        self._goal_sent = True
        self.get_logger().info(
            f'Sending batch goal for {self._batch_size} objects')
        pending_goal = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        pending_goal.add_done_callback(self._goal_response_callback)

    def _feedback_callback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.get_logger().info(
            f'{feedback.status}: {feedback.completed_objects} objects complete '
            f'({feedback.progress:.0%})')

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Batch goal was rejected')
            return
        self.get_logger().info('Batch goal was accepted')
        pending_result = goal_handle.get_result_async()
        pending_result.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        result = future.result().result
        self.get_logger().info(result.message)
        for object_result in result.object_results:
            self.get_logger().info(
                f'Object {object_result.object_id}: success={object_result.success}, '
                f'candidates={len(object_result.candidates)}')


def main(args=None) -> None:
    """Run the batch action client."""
    rclpy.init(args=args)
    node = GraspGenXBatchClient()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
