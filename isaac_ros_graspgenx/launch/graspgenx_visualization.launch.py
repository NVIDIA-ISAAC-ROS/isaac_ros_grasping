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

"""Launch persistent RViz visualization for a GraspGen-X PFC run."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Generate the GraspGen-X visualization launch description."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    gripper_name = LaunchConfiguration('gripper_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the simulation clock if true.'),
        DeclareLaunchArgument(
            'gripper_name',
            default_value='robotiq_2f_140',
            description='Gripper model to render for each grasp candidate.'),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz if true.'),
        DeclareLaunchArgument(
            'full_cloud_stride',
            default_value='4',
            description='Pixel stride used to generate the full-scene point cloud.'),
        DeclareLaunchArgument(
            'request_grasps',
            default_value='false',
            description='Request grasps from the GraspGen-X action server if true.'),
        DeclareLaunchArgument(
            'batch_mode',
            default_value='false',
            description='Request grasps for batches of segmented object clouds if true.'),
        DeclareLaunchArgument(
            'selected_grasp_topic',
            default_value='/graspgenx/selected_grasp',
            description='Topic from which to visualize the selected grasp.'),
        Node(
            package='isaac_ros_graspgenx',
            executable='graspgenx_visualization',
            name='graspgenx_visualization',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'gripper_name': gripper_name,
                'full_cloud_stride': ParameterValue(
                    LaunchConfiguration('full_cloud_stride'), value_type=int),
                'request_grasps': ParameterValue(
                    LaunchConfiguration('request_grasps'), value_type=bool),
                'batch_mode': ParameterValue(
                    LaunchConfiguration('batch_mode'), value_type=bool),
                'selected_grasp_topic': LaunchConfiguration('selected_grasp_topic'),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='graspgenx_rviz',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
        ),
    ])
