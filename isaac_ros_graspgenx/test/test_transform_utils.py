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

"""Unit tests for transform utilities."""

from geometry_msgs.msg import Pose
import numpy as np
import pytest
from utils.transform_utils import (
    pose_to_matrix,
    transform_points,
)


def test_pose_to_matrix_preserves_translation():
    pose = Pose()
    pose.position.x = 1.0
    pose.position.y = 2.0
    pose.position.z = 3.0
    pose.orientation.w = 1.0
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(pose_to_matrix(pose), expected)


def test_transform_points_identity():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    transformed = transform_points(points, np.eye(4))
    assert np.allclose(transformed, points)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
