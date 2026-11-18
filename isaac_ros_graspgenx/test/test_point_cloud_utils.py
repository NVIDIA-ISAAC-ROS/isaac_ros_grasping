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

"""Unit tests for point cloud utilities."""

from builtin_interfaces.msg import Time
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
from utils.point_cloud_utils import (
    depth_to_point_cloud,
    numpy_to_point_cloud2,
    point_cloud2_to_numpy,
)


def _make_depth_image(depth_mm: np.ndarray, encoding: str = '16UC1') -> Image:
    image = Image()
    image.height = depth_mm.shape[0]
    image.width = depth_mm.shape[1]
    image.encoding = encoding
    image.step = depth_mm.shape[1] * depth_mm.dtype.itemsize
    image.header.frame_id = 'camera_optical'
    image.data = depth_mm.tobytes()
    return image


def _make_camera_info(width: int, height: int) -> CameraInfo:
    info = CameraInfo()
    info.width = width
    info.height = height
    info.k = [500.0, 0.0, width / 2.0, 0.0, 500.0, height / 2.0, 0.0, 0.0, 1.0]
    return info


def test_depth_to_point_cloud_applies_stride():
    depth = np.full((4, 4), 1000, dtype=np.uint16)
    points, frame_id = depth_to_point_cloud(
        _make_depth_image(depth),
        _make_camera_info(4, 4),
        stride=2,
    )
    assert frame_id == 'camera_optical'
    assert points.shape == (4, 3)
    assert np.allclose(points[:, 2], 1.0)


def test_point_cloud2_round_trip():
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    cloud = numpy_to_point_cloud2(points, 'base_link', Time(sec=1, nanosec=0))
    recovered = point_cloud2_to_numpy(cloud)
    assert recovered.shape == (2, 3)
    assert np.allclose(recovered, points)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
