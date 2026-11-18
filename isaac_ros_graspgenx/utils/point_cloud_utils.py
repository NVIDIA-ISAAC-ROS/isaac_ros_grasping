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

"""Point cloud conversion helpers."""

from typing import Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _depth_image_to_meters(depth_image: Image) -> np.ndarray:
    """Decode a ROS depth image into a float32 array in meters."""
    if depth_image.encoding == '16UC1':
        depth = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(
            depth_image.height, depth_image.width
        )
        return depth.astype(np.float32) / 1000.0
    if depth_image.encoding == '32FC1':
        return np.frombuffer(depth_image.data, dtype=np.float32).reshape(
            depth_image.height, depth_image.width
        )
    raise ValueError(
        f'Unsupported depth encoding {depth_image.encoding}; expected 16UC1 or 32FC1'
    )


def depth_to_point_cloud(
    depth_image: Image,
    camera_info: CameraInfo,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    stride: int = 4,
) -> Tuple[np.ndarray, str]:
    """Back-project a strided registered-depth image into an Nx3 point cloud."""
    if stride < 1:
        raise ValueError(f'stride must be positive, got {stride}')

    depth_m = _depth_image_to_meters(depth_image)
    ys = np.arange(0, depth_image.height, stride, dtype=np.int32)
    xs = np.arange(0, depth_image.width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled_depth = depth_m[grid_y, grid_x]
    valid = np.isfinite(sampled_depth)
    valid &= (sampled_depth >= min_depth_m) & (sampled_depth <= max_depth_m)

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), depth_image.header.frame_id

    z = sampled_depth[valid]
    pixel_x = grid_x[valid].astype(np.float32)
    pixel_y = grid_y[valid].astype(np.float32)
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])
    x = (pixel_x - cx) * z / fx
    y = (pixel_y - cy) * z / fy
    return (
        np.stack([x, y, z], axis=1).astype(np.float32),
        depth_image.header.frame_id,
    )


def numpy_to_point_cloud2(
    points: np.ndarray,
    frame_id: str,
    stamp,
) -> PointCloud2:
    """Pack an Nx3 numpy array into a sensor_msgs PointCloud2."""
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float32)
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp
    return point_cloud2.create_cloud_xyz32(header, points.tolist())


def point_cloud2_to_numpy(cloud: PointCloud2) -> np.ndarray:
    """Extract Nx3 points from a sensor_msgs PointCloud2."""
    if cloud.width == 0 or cloud.height == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = point_cloud2.read_points_numpy(
        cloud,
        field_names=('x', 'y', 'z'),
        skip_nans=True,
    )
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)
