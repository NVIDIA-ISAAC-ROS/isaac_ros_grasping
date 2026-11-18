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

"""Unit tests for GraspGen-X RViz geometry."""

from unittest.mock import Mock

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseArray
from isaac_ros_graspgenx_interfaces.action import GenerateGrasps, GenerateGraspsBatch
from isaac_ros_graspgenx_interfaces.msg import (
    GraspCandidate,
    ObjectGraspResult,
    SegmentedObjectCloudBatch,
)
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from utils.graspgenx_visualization import (
    _gripper_line_points,
    _group_color,
    GraspGenXVisualization,
)
from utils.point_cloud_utils import numpy_to_point_cloud2, point_cloud2_to_numpy
from visualization_msgs.msg import Marker


def test_gripper_wireframe_has_four_line_segments():
    points = _gripper_line_points(np.eye(4), 'robotiq_2f_140')
    assert len(points) == 8


def test_gripper_fingers_close_along_local_x():
    points = _gripper_line_points(np.eye(4), 'robotiq_2f_140')
    finger_base_left, finger_base_right = points[2], points[3]
    assert finger_base_left.x == pytest.approx(-0.07)
    assert finger_base_right.x == pytest.approx(0.07)
    assert finger_base_left.y == finger_base_right.y == 0.0


def test_gripper_wireframe_follows_pose_translation():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    points = _gripper_line_points(pose, 'robotiq_2f_140')
    assert points[0].x == 1.0
    assert points[0].y == 2.0
    assert points[0].z == 3.0


def test_group_colors_are_stable_and_distinct():
    assert _group_color(7) == _group_color(7)
    assert _group_color(7) != _group_color(8)
    assert all(0.0 <= channel <= 1.0 for channel in _group_color(7))


@pytest.fixture
def visualization():
    rclpy.init()
    node = GraspGenXVisualization()
    node._grasp_marker_pub = Mock()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('batch_mode', [False, True])
@pytest.mark.parametrize('topic_first', [False, True])
def test_requested_results_render_once(visualization, batch_mode, topic_first):
    visualization.set_parameters([
        Parameter('request_grasps', value=True),
        Parameter('batch_mode', value=batch_mode),
    ])
    candidate = GraspCandidate()
    candidate.pose.header.frame_id = 'base_link'
    candidate.pose.pose.orientation.w = 1.0
    candidate.confidence = 0.8
    objects = [ObjectGraspResult(success=True, candidates=[candidate])]
    if batch_mode:
        objects.append(ObjectGraspResult(success=True, candidates=[candidate]))
        result = GenerateGraspsBatch.Result(object_results=objects, all_objects_succeeded=True)
        callback = visualization._batch_grasp_result_callback
    else:
        result = GenerateGrasps.Result(object_result=objects[0])
        callback = visualization._grasp_result_callback
    topic = PoseArray(poses=[candidate.pose.pose for _ in objects])
    future = Mock()
    future.result.return_value.result = result

    if topic_first:
        visualization._candidate_pose_callback(topic)
    callback(future)
    if not topic_first:
        visualization._candidate_pose_callback(topic)

    visualization._grasp_marker_pub.publish.assert_called_once()
    message = visualization._grasp_marker_pub.publish.call_args.args[0]
    markers = [marker for marker in message.markers if marker.action == Marker.ADD]
    assert len(markers) == len(objects)
    expected_namespaces = {'graspgenx_grasps_1', 'graspgenx_grasps_2'} if batch_mode else {
        'graspgenx_grasps'}
    assert {marker.ns for marker in markers} == expected_namespaces


def test_passive_visualization_renders_candidate_topic(visualization):
    candidate = GraspCandidate()
    candidate.pose.pose.orientation.w = 1.0
    visualization._candidate_pose_callback(PoseArray(poses=[candidate.pose.pose]))

    visualization._grasp_marker_pub.publish.assert_called_once()
    message = visualization._grasp_marker_pub.publish.call_args.args[0]
    markers = [marker for marker in message.markers if marker.action == Marker.ADD]
    assert len(markers) == 1
    assert markers[0].ns == 'graspgenx_grasps'


def test_segmented_batch_requests_latest_batch(visualization):
    visualization.set_parameters([
        Parameter('request_grasps', value=True),
        Parameter('batch_mode', value=True),
    ])
    visualization._send_batch_grasp_goal = Mock()

    visualization._segmented_cloud_batch_callback(
        SegmentedObjectCloudBatch(clouds=[PointCloud2()]))

    visualization._send_batch_grasp_goal.assert_called_once_with()


def test_segmented_batch_shows_every_object(visualization):
    visualization._segmented_cloud_pub = Mock()
    visualization._mean_pub = Mock()
    first = np.array([[0.0, 0.0, 1.0], [0.0, 0.2, 1.0]], dtype=np.float32)
    second = np.array([[0.5, 0.0, 2.0]], dtype=np.float32)

    visualization._segmented_cloud_batch_callback(SegmentedObjectCloudBatch(clouds=[
        numpy_to_point_cloud2(first, 'camera', Time()),
        numpy_to_point_cloud2(second, 'camera', Time()),
    ]))

    cloud = visualization._segmented_cloud_pub.publish.call_args.args[0]
    assert point_cloud2_to_numpy(cloud).shape == (3, 3)
    message = visualization._mean_pub.publish.call_args.args[0]
    means = [marker.pose.position for marker in message.markers if marker.action == Marker.ADD]
    assert [mean.y for mean in means] == pytest.approx([0.1, 0.0])
    assert [mean.z for mean in means] == pytest.approx([1.0, 2.0])


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
