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

"""
Proof-of-life test for the Isaac ROS GraspGenX quickstart.

1. Launches the quickstart without RViz and replays the included can rosbag.
2. Waits for the segmented object cloud and sends one GenerateGrasps goal.
3. Verifies that the result contains ranked, normalized grasps near the object.
"""

import math
import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_graspgenx_interfaces.action import GenerateGrasps
from isaac_ros_test import IsaacROSBaseTest
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from utils.point_cloud_utils import point_cloud2_to_numpy

ROSBAG_PATH = pathlib.Path(__file__).parent / 'rosbags' / 'foundationstereo_sam_can'
ORIGIN_FRAME = 'camera_2_infra1_optical_frame'


@pytest.mark.rostest
def generate_test_description():
    """Launch the quickstart on the can rosbag with the installed GraspGenX engine."""
    engine_file_path = os.environ.get('GRASPGENX_ENGINE_PATH', '')
    custom_plugin_lib = os.environ.get('GRASPGENX_PLUGIN_PATH', '')
    for name, path in (('GRASPGENX_ENGINE_PATH', engine_file_path),
                       ('GRASPGENX_PLUGIN_PATH', custom_plugin_lib)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'{name}={path!r} is not a file. Run install_graspgenx_models.sh from '
                'isaac_ros_graspgenx_models_install and export the generated engine and '
                'plugin paths before running this test.')

    quickstart_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('isaac_ros_graspgenx'),
            'launch', 'graspgenx_quickstart.launch.py')),
        launch_arguments={
            'engine_file_path': engine_file_path,
            'custom_plugin_lib': custom_plugin_lib,
            'origin_frame': ORIGIN_FRAME,
            'depth_topic': '/offline_foundationstereo/depth_image',
            'camera_info_topic': '/offline_foundationstereo/model_camera_info',
            'rgb_topic': '/offline_foundationstereo/model_rgb',
            'mask_topic': '/segmentation_can',
            'launch_rviz': 'false',
            'request_grasps': 'false',
        }.items(),
    )

    rosbag_play = launch.actions.ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '--loop', str(ROSBAG_PATH)],
        output='screen',
    )

    return IsaacROSGraspGenXPOLTest.generate_test_description([
        quickstart_launch,
        rosbag_play,
    ])


class IsaacROSGraspGenXPOLTest(IsaacROSBaseTest):
    """Generate grasps for the can in the sample rosbag and check the returned poses."""

    # Covers TensorRT engine deserialization on the first request.
    TIMEOUT_SEC = 300.0
    OBJECT_ID = 1
    NUM_GRASPS = 50
    TOP_K_GRASPS = 20
    # The robotiq_2f_140 fingertips are 0.18 m along +Z from the grasp origin.
    MAX_GRASP_DISTANCE_M = 0.3

    def test_generate_grasps_for_sample_object(self) -> None:
        clouds = []
        cloud_sub = self.node.create_subscription(
            PointCloud2, '/graspgenx/segmented_object_cloud', clouds.append,
            qos_profile_sensor_data)
        grasp_client = ActionClient(self.node, GenerateGrasps, '/graspgenx/generate_grasps')
        try:
            end_time = time.time() + self.TIMEOUT_SEC
            while not clouds and time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.assertTrue(
                clouds, f'No cloud on /graspgenx/segmented_object_cloud from {ROSBAG_PATH}')
            object_points = point_cloud2_to_numpy(clouds[-1])
            self.assertGreater(len(object_points), 0, 'Segmented object cloud has no points')
            object_center = object_points.mean(axis=0)

            self.assertTrue(
                grasp_client.wait_for_server(timeout_sec=max(0.0, end_time - time.time())),
                '/graspgenx/generate_grasps action server is unavailable')
            goal = GenerateGrasps.Goal()
            goal.object.object_id = self.OBJECT_ID
            goal.num_grasps = self.NUM_GRASPS
            goal.grasp_confidence_threshold = -1.0
            goal.top_k_grasps = self.TOP_K_GRASPS
            goal_future = grasp_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(
                self.node, goal_future, timeout_sec=max(0.0, end_time - time.time()))
            self.assertTrue(goal_future.done(), 'Timed out waiting for goal acceptance')
            goal_handle = goal_future.result()
            self.assertTrue(goal_handle.accepted, 'GraspGenX rejected the grasp goal')

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(
                self.node, result_future, timeout_sec=max(0.0, end_time - time.time()))
            self.assertTrue(result_future.done(), 'Timed out waiting for the grasp result')
            result = result_future.result().result
            object_result = result.object_result
            self.assertTrue(
                object_result.success, f'Grasp generation failed: {result.message}')
            self.assertEqual(object_result.object_id, self.OBJECT_ID)

            candidates = object_result.candidates
            self.assertGreater(len(candidates), 0, 'Grasp result has no candidates')
            self.assertLessEqual(len(candidates), self.TOP_K_GRASPS)
            confidences = [candidate.confidence for candidate in candidates]
            self.assertEqual(
                confidences, sorted(confidences, reverse=True),
                'Candidates are not in descending confidence order')
            for index, candidate in enumerate(candidates):
                pose = candidate.pose
                self.assertEqual(pose.header.frame_id, ORIGIN_FRAME)
                self.assertTrue(
                    0.0 <= candidate.confidence <= 1.0,
                    f'Candidate {index} confidence {candidate.confidence} is outside [0, 1]')
                orientation = pose.pose.orientation
                self.assertAlmostEqual(
                    math.sqrt(orientation.x ** 2 + orientation.y ** 2 +
                              orientation.z ** 2 + orientation.w ** 2),
                    1.0, delta=1e-3, msg=f'Candidate {index} quaternion is not normalized')
                position = np.array(
                    [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                distance = float(np.linalg.norm(position - object_center))
                self.assertLess(
                    distance, self.MAX_GRASP_DISTANCE_M,
                    f'Candidate {index} at {position.tolist()} is {distance:.3f} m from the '
                    f'object center {object_center.tolist()}')

            self.node.get_logger().info(
                f'Received {len(candidates)} grasps; best confidence {confidences[0]:.3f}')
        finally:
            grasp_client.destroy()
            self.node.destroy_subscription(cloud_sub)
