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

"""Pose and point-cloud transform helpers for GraspGen-X ROS integration."""

from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a geometry_msgs Pose to a 4x4 homogeneous matrix."""
    rotation = Rotation.from_quat([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    if points.size == 0:
        return points.reshape(0, 3)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points.astype(np.float64), ones])
    transformed = (transform_matrix @ homogeneous.T).T
    return transformed[:, :3].astype(np.float32)
