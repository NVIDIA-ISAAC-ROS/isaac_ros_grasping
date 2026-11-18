#!/usr/bin/env python3

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

"""Publish persistent RViz views of scene geometry and GraspGen-X candidates."""

from __future__ import annotations

import colorsys
import math
import time

from geometry_msgs.msg import Point, PoseArray, PoseStamped
from isaac_ros_graspgenx_interfaces.action import (
    GenerateGrasps,
    GenerateGraspsBatch,
)
from isaac_ros_graspgenx_interfaces.msg import (
    SegmentedObjectCloudBatch,
)
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from utils.point_cloud_utils import (
    depth_to_point_cloud,
    numpy_to_point_cloud2,
    point_cloud2_to_numpy,
)
from utils.transform_utils import (
    pose_to_matrix,
    transform_points,
)
from visualization_msgs.msg import Marker, MarkerArray


_COMPATIBILITY_OBJECT_ID = 1


def _gripper_dimensions(gripper_name: str) -> tuple[float, float, float]:
    """Return width, palm depth, and finger length for RViz wireframes."""
    dimensions = {
        'robotiq_2f_140': (0.14, 0.10, 0.08),
        'robotiq_2f_85': (0.085, 0.085, 0.06),
        'grav': (0.08, 0.07, 0.05),
    }
    return dimensions.get(gripper_name, (0.08, 0.07, 0.05))


def _group_color(group_index: int) -> tuple[float, float, float]:
    """Return a stable, distinguishable RGB color for a batch position."""
    hue = (int(group_index) * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.72, 0.95)


def _gripper_line_points(
    pose_matrix: np.ndarray,
    gripper_name: str,
) -> list[Point]:
    """Create transformed line-list points for a parallel-jaw gripper."""
    width, palm_depth, finger_length = _gripper_dimensions(gripper_name)
    half_width = width / 2.0
    tip_z = palm_depth + finger_length
    # GraspGen grasp frames approach along +Z and close the fingers along X.
    local_vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, palm_depth],
        [-half_width, 0.0, palm_depth],
        [half_width, 0.0, palm_depth],
        [-half_width, 0.0, tip_z],
        [half_width, 0.0, tip_z],
    ], dtype=np.float32)
    line_indices = [(0, 1), (2, 3), (2, 4), (3, 5)]
    world_vertices = transform_points(local_vertices, pose_matrix)

    points: list[Point] = []
    for start_idx, end_idx in line_indices:
        for vertex_idx in (start_idx, end_idx):
            vertex = world_vertices[vertex_idx]
            points.append(Point(
                x=float(vertex[0]),
                y=float(vertex[1]),
                z=float(vertex[2]),
            ))
    return points


class GraspGenXVisualization(Node):
    """Visualize full and segmented clouds, object mean, and ranked grasps."""

    def __init__(self) -> None:
        super().__init__('graspgenx_visualization')
        self.declare_parameter(
            'depth_topic', '/front_stereo_camera/depth/ground_truth')
        self.declare_parameter(
            'camera_info_topic', '/front_stereo_camera/depth/camera_info')
        self.declare_parameter(
            'segmented_cloud_topic', '/graspgenx/segmented_object_cloud')
        self.declare_parameter('full_cloud_stride', 4)
        self.declare_parameter('full_cloud_publish_period_sec', 1.0)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('origin_frame', 'base_link')
        self.declare_parameter('gripper_name', 'robotiq_2f_140')
        self.declare_parameter('num_grasps', 50)
        self.declare_parameter('top_k_grasps', 20)
        self.declare_parameter('grasp_confidence_threshold', -1.0)
        self.declare_parameter('request_grasps', False)
        self.declare_parameter('batch_mode', False)
        self.declare_parameter('selected_grasp_topic', '/graspgenx/selected_grasp')

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._full_cloud_pub = self.create_publisher(
            PointCloud2, '/graspgenx/visualization/full_scene_cloud', latched_qos)
        self._segmented_cloud_pub = self.create_publisher(
            PointCloud2, '/graspgenx/visualization/segmented_object_cloud', latched_qos)
        self._mean_pub = self.create_publisher(
            MarkerArray, '/graspgenx/visualization/object_mean', latched_qos)
        self._grasp_marker_pub = self.create_publisher(
            MarkerArray, '/graspgenx/visualization/grasp_markers', latched_qos)
        self._selected_grasp_marker_pub = self.create_publisher(
            MarkerArray, '/graspgenx/visualization/selected_grasp_marker', latched_qos)

        self._camera_info: CameraInfo | None = None
        self._last_full_cloud_publish_time = -math.inf
        self._goal_sent = False
        self._batch_marker_counts: dict[int, int] = {}
        self._grasp_client = ActionClient(
            self, GenerateGrasps, 'graspgenx/generate_grasps')
        self._batch_grasp_client = ActionClient(
            self, GenerateGraspsBatch, 'graspgenx/generate_grasps_batch')

        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_callback,
            10,
        )
        self._depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self._depth_callback,
            10,
        )
        self._segmented_cloud_sub = self.create_subscription(
            PointCloud2,
            self.get_parameter('segmented_cloud_topic').value,
            self._segmented_cloud_callback,
            10,
        )
        self._segmented_cloud_batch_sub = self.create_subscription(
            SegmentedObjectCloudBatch,
            '/graspgenx/segmented_object_cloud_batch',
            self._segmented_cloud_batch_callback,
            10,
        )
        self._candidate_sub = self.create_subscription(
            PoseArray,
            '/graspgenx/grasp_candidates',
            self._candidate_pose_callback,
            latched_qos,
        )
        self._selected_grasp_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('selected_grasp_topic').value,
            self._selected_grasp_callback,
            latched_qos,
        )

    def _camera_info_callback(self, camera_info: CameraInfo) -> None:
        self._camera_info = camera_info

    def _depth_callback(self, depth_image: Image) -> None:
        if self._camera_info is None:
            return
        now = time.monotonic()
        publish_period = float(
            self.get_parameter('full_cloud_publish_period_sec').value)
        if now - self._last_full_cloud_publish_time < publish_period:
            return
        try:
            points, frame_id = depth_to_point_cloud(
                depth_image,
                self._camera_info,
                min_depth_m=float(self.get_parameter('min_depth_m').value),
                max_depth_m=float(self.get_parameter('max_depth_m').value),
                stride=int(self.get_parameter('full_cloud_stride').value),
            )
        except ValueError as exc:
            self.get_logger().warning(f'Cannot visualize full cloud: {exc}')
            return
        if points.size == 0:
            return

        cloud = numpy_to_point_cloud2(
            points, frame_id, depth_image.header.stamp)
        self._full_cloud_pub.publish(cloud)
        self._last_full_cloud_publish_time = now
        self.get_logger().info(
            f'Published full scene visualization cloud with {points.shape[0]} points')

    def _segmented_cloud_callback(self, cloud: PointCloud2) -> None:
        self._segmented_cloud_pub.publish(cloud)
        points = point_cloud2_to_numpy(cloud)
        if points.size != 0:
            self._publish_means(cloud.header, [points.mean(axis=0)])
        if (
            self.get_parameter('request_grasps').value
            and not self.get_parameter('batch_mode').value
            and not self._goal_sent
        ):
            self._send_grasp_goal()

    def _segmented_cloud_batch_callback(
        self, batch: SegmentedObjectCloudBatch
    ) -> None:
        object_points = [
            points for points in map(point_cloud2_to_numpy, batch.clouds) if points.size != 0]
        if object_points:
            header = batch.clouds[0].header
            self._segmented_cloud_pub.publish(numpy_to_point_cloud2(
                np.concatenate(object_points), header.frame_id, header.stamp))
            self._publish_means(header, [points.mean(axis=0) for points in object_points])
        if (
            batch.clouds
            and self.get_parameter('request_grasps').value
            and self.get_parameter('batch_mode').value
            and not self._goal_sent
        ):
            self._send_batch_grasp_goal()

    def _candidate_pose_callback(self, pose_array: PoseArray) -> None:
        # Requested grasps are rendered from action results, preserving scores and object groups.
        if self.get_parameter('request_grasps').value:
            return
        count = len(pose_array.poses)
        if count == 0:
            return
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers = [delete_all]
        gripper_name = self.get_parameter('gripper_name').value
        denominator = max(count - 1, 1)
        for marker_id, pose in enumerate(pose_array.poses):
            normalized = 1.0 - marker_id / denominator
            marker = Marker()
            marker.header = pose_array.header
            marker.ns = 'graspgenx_grasps'
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.006
            marker.color.r = float(1.0 - normalized)
            marker.color.g = float(normalized)
            marker.color.b = 0.15
            marker.color.a = 0.95
            marker.points = _gripper_line_points(
                pose_to_matrix(pose), gripper_name)
            markers.append(marker)
        self._grasp_marker_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info(
            f'Published {count} ranked grasp wireframes')

    def _selected_grasp_callback(self, selected: PoseStamped) -> None:
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL

        wireframe = Marker()
        wireframe.header = selected.header
        wireframe.ns = 'graspgenx_selected_grasp'
        wireframe.id = 0
        wireframe.type = Marker.LINE_LIST
        wireframe.action = Marker.ADD
        wireframe.pose.orientation.w = 1.0
        wireframe.scale.x = 0.014
        wireframe.color.r = 0.1
        wireframe.color.g = 0.75
        wireframe.color.b = 1.0
        wireframe.color.a = 1.0
        wireframe.points = _gripper_line_points(
            pose_to_matrix(selected.pose),
            self.get_parameter('gripper_name').value,
        )

        label = Marker()
        label.header = selected.header
        label.ns = 'graspgenx_selected_grasp'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = selected.pose.position.x
        label.pose.position.y = selected.pose.position.y
        label.pose.position.z = selected.pose.position.z + 0.12
        label.pose.orientation.w = 1.0
        label.scale.z = 0.04
        label.color.r = 0.1
        label.color.g = 0.75
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = 'cuMotion selected grasp'

        self._selected_grasp_marker_pub.publish(
            MarkerArray(markers=[delete_all, wireframe, label]))
        self.get_logger().info('Highlighted the cuMotion-selected grasp')

    def _publish_means(self, header: Header, means: list[np.ndarray]) -> None:
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers = [delete_all]
        for marker_id, mean in enumerate(means):
            sphere = Marker()
            sphere.header = header
            sphere.ns = 'graspgenx_object_mean'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(
                x=float(mean[0]), y=float(mean[1]), z=float(mean[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.04
            sphere.scale.y = 0.04
            sphere.scale.z = 0.04
            sphere.color.r = 1.0
            sphere.color.g = 0.85
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            markers.append(sphere)

        self._mean_pub.publish(MarkerArray(markers=markers))

    def _send_grasp_goal(self) -> None:
        if not self._grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning('GraspGen-X action server is unavailable')
            return
        self._goal_sent = True
        goal = GenerateGrasps.Goal()
        goal.object.object_id = _COMPATIBILITY_OBJECT_ID
        goal.gripper_name = self.get_parameter('gripper_name').value
        goal.num_grasps = int(self.get_parameter('num_grasps').value)
        goal.top_k_grasps = int(self.get_parameter('top_k_grasps').value)
        goal.grasp_confidence_threshold = float(
            self.get_parameter('grasp_confidence_threshold').value)
        future = self._grasp_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _send_batch_grasp_goal(self) -> None:
        if not self._batch_grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning('GraspGen-X batch action server is unavailable')
            return
        self._goal_sent = True
        goal = GenerateGraspsBatch.Goal()
        goal.gripper_name = self.get_parameter('gripper_name').value
        goal.num_grasps = int(self.get_parameter('num_grasps').value)
        goal.top_k_grasps = int(self.get_parameter('top_k_grasps').value)
        goal.grasp_confidence_threshold = float(
            self.get_parameter('grasp_confidence_threshold').value)
        goal.random_seed = 0
        future = self._batch_grasp_client.send_goal_async(goal)
        future.add_done_callback(self._batch_goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Visualization grasp goal was rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._grasp_result_callback)

    def _grasp_result_callback(self, future) -> None:
        result = future.result().result
        if not result.object_result.success or not result.object_result.candidates:
            self.get_logger().error(
                result.message or 'No grasp candidates to visualize')
            return
        self._publish_grasps(result.object_result)

    def _batch_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Visualization batch grasp goal was rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._batch_grasp_result_callback)

    def _batch_grasp_result_callback(self, future) -> None:
        result = future.result().result
        self._publish_batch_grasps(result.object_results)
        if not result.all_objects_succeeded:
            self.get_logger().warning(result.message)

    def _publish_grasps(self, result) -> None:
        confidences = np.asarray(
            [candidate.confidence for candidate in result.candidates],
            dtype=np.float64,
        )
        finite = confidences[np.isfinite(confidences)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 1.0
        span = max(high - low, 1e-9)
        gripper_name = self.get_parameter('gripper_name').value

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers = [delete_all]
        for marker_id, candidate in enumerate(result.candidates):
            confidence = float(candidate.confidence)
            normalized = (
                min(max((confidence - low) / span, 0.0), 1.0)
                if math.isfinite(confidence) else 0.0)
            marker = Marker()
            marker.header = candidate.pose.header
            marker.ns = 'graspgenx_grasps'
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.006
            marker.color.r = float(1.0 - normalized)
            marker.color.g = float(normalized)
            marker.color.b = 0.15
            marker.color.a = 0.95
            marker.points = _gripper_line_points(
                pose_to_matrix(candidate.pose.pose), gripper_name)
            markers.append(marker)

        self._grasp_marker_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info(
            f'Published {len(result.candidates)} confidence-colored grasp wireframes')

    def _publish_batch_grasps(self, object_results) -> None:
        markers = []
        for group_index, count in self._batch_marker_counts.items():
            for marker_id in range(count):
                marker = Marker()
                marker.ns = f'graspgenx_grasps_{group_index}'
                marker.id = marker_id
                marker.action = Marker.DELETE
                markers.append(marker)

        new_counts = {}
        gripper_name = self.get_parameter('gripper_name').value
        for group_index, object_result in enumerate(object_results, start=1):
            candidates = object_result.candidates if object_result.success else []
            new_counts[group_index] = len(candidates)
            red, green, blue = _group_color(group_index)
            for marker_id, candidate in enumerate(candidates):
                marker = Marker()
                marker.header = candidate.pose.header
                marker.ns = f'graspgenx_grasps_{group_index}'
                marker.id = marker_id
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.006
                marker.color.r = red
                marker.color.g = green
                marker.color.b = blue
                marker.color.a = 0.95
                marker.points = _gripper_line_points(
                    pose_to_matrix(candidate.pose.pose), gripper_name)
                markers.append(marker)
        self._batch_marker_counts = new_counts
        if markers:
            self._grasp_marker_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info(
            f'Published grouped grasp wireframes for {len(new_counts)} objects')


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenXVisualization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
