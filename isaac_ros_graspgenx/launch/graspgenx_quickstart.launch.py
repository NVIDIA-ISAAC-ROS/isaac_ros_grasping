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

"""Launch standalone grasp generation from registered depth and one or more object masks."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import yaml


def load_batch_mask_topics(config_file):
    """Return the per-object mask topics listed in a batch config file."""
    with open(config_file, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    mask_topics = config.get('mask_topics') if isinstance(config, dict) else None
    if (
        not isinstance(mask_topics, list)
        or not mask_topics
        or not all(isinstance(topic, str) and topic for topic in mask_topics)
    ):
        raise RuntimeError(
            f"{config_file} must define 'mask_topics' as a non-empty list of topic names, "
            f'got {mask_topics!r}')
    return mask_topics


def launch_setup(context):
    """Compose cloud builders, the action server, and TensorRT with optional RViz."""
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    origin_frame = LaunchConfiguration('origin_frame')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    batch_mode = LaunchConfiguration('batch_mode').perform(context).lower() == 'true'
    if batch_mode:
        mask_topics = load_batch_mask_topics(
            LaunchConfiguration('batch_config_file').perform(context))
    else:
        mask_topics = [LaunchConfiguration('mask_topic').perform(context)]

    # In batch mode each object gets its own cloud topic for GenerateBatchInputNode to group.
    cloud_topics = (
        [f'/graspgenx/object_cloud_{index}' for index in range(len(mask_topics))]
        if batch_mode else ['/graspgenx/segmented_object_cloud'])
    cloud_components = [
        ComposableNode(
            package='isaac_ros_depth_image_proc',
            plugin='nvidia::isaac_ros::depth_image_proc::PointCloudXYZMaskedNode',
            name='graspgenx_segmented_point_cloud_builder' + (f'_{index}' if index else ''),
            parameters=[{
                'use_sim_time': use_sim_time,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'segmentation_mask_topic': mask_topic,
                'input_qos': 'DEFAULT',
                'output_topic': cloud_topic,
                'min_depth_m': 0.05,
                'max_depth_m': 2.0,
                'sync_slop_sec': 0.5,
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        )
        for index, (mask_topic, cloud_topic) in enumerate(zip(mask_topics, cloud_topics))
    ]
    if batch_mode:
        cloud_components.append(ComposableNode(
            package='isaac_ros_graspgenx',
            plugin='nvidia::isaac_ros::graspgenx::GenerateBatchInputNode',
            name='graspgenx_generate_batch_input',
            parameters=[{
                'use_sim_time': use_sim_time,
                'input_topics': cloud_topics,
            }],
            extra_arguments=[{'use_intra_process_comms': True}],
        ))

    return [
        ComposableNodeContainer(
            name='graspgenx_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            output='screen',
            composable_node_descriptions=[
                ComposableNode(
                    package='isaac_ros_tensor_rt',
                    plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                    name='graspgenx_tensor_rt',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('isaac_ros_graspgenx'),
                            'config', 'graspgenx_tensor_rt.yaml']),
                        {
                            'use_sim_time': use_sim_time,
                            'engine_file_path': LaunchConfiguration('engine_file_path'),
                            'custom_plugin_lib': LaunchConfiguration('custom_plugin_lib'),
                        },
                    ],
                    remappings=[
                        ('tensor_pub', 'graspgenx/inference/input'),
                        ('tensor_sub', 'graspgenx/inference/output'),
                    ],
                    extra_arguments=[{'use_intra_process_comms': True}],
                ),
                *cloud_components,
                ComposableNode(
                    package='isaac_ros_graspgenx',
                    plugin='nvidia::isaac_ros::graspgenx::GenerateGraspNode',
                    name='graspgenx_generate_grasp_node',
                    extra_arguments=[{'use_intra_process_comms': True}],
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'inference_timeout_sec': ParameterValue(
                            LaunchConfiguration('inference_timeout_sec'), value_type=float),
                        'origin_frame': origin_frame,
                        'default_gripper_name': 'robotiq_2f_140',
                    }],
                ),
            ],
        ),
        Node(
            package='isaac_ros_graspgenx',
            executable='graspgenx_visualization',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'origin_frame': origin_frame,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'gripper_name': 'robotiq_2f_140',
                'request_grasps': ParameterValue(
                    LaunchConfiguration('request_grasps'), value_type=bool),
                'batch_mode': batch_mode,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=[
                '-d', PathJoinSubstitution([
                    FindPackageShare('isaac_ros_graspgenx'),
                    'rviz', 'graspgenx_quickstart.rviz']),
                '-f', origin_frame,
            ],
            remappings=[
                ('/graspgenx/rgb_image', LaunchConfiguration('rgb_topic')),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
        ),
    ]


def generate_launch_description():
    """Declare quickstart arguments; nodes depend on the resolved mask topics."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'engine_file_path',
            default_value=EnvironmentVariable('GRASPGENX_ENGINE_PATH', default_value=''),
            description='TensorRT engine built for the current GPU and runtime'),
        DeclareLaunchArgument(
            'custom_plugin_lib',
            default_value=EnvironmentVariable('GRASPGENX_PLUGIN_PATH', default_value=''),
            description='libgraspgenx_plugins.so installed alongside the engine'),
        DeclareLaunchArgument('inference_timeout_sec', default_value='60.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('request_grasps', default_value='true'),
        DeclareLaunchArgument('origin_frame', default_value='camera_depth_optical_frame'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_rect_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/depth/camera_info'),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/camera/color/image_raw',
            description='RGB image shown in the RViz RGB Scene panel'),
        DeclareLaunchArgument(
            'mask_topic',
            default_value='/segmentation_mask',
            description='Object mask topic for single-object mode'),
        DeclareLaunchArgument(
            'batch_mode',
            default_value='false',
            description='Request grasps for every mask topic in batch_config_file from one '
                        'frame in a single batch'),
        DeclareLaunchArgument(
            'batch_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('isaac_ros_graspgenx'),
                'config', 'graspgenx_quickstart_batch.yaml']),
            description='YAML file with a mask_topics list, one mask topic per object; '
                        'used only with batch_mode:=true'),
        OpaqueFunction(function=launch_setup),
    ])
