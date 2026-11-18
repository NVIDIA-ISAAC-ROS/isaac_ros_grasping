# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference to an object cloud owned by the GraspGenX server.
# Valid until the cloud is evicted from the server's cache or the server restarts.

# Cloud ID published by the server on /graspgenx/available_object_cloud.
uint64 object_id

# Original cloud frame and acquisition time. This is informational metadata;
# the server uses object_id to look up the retained cloud and its original header.
std_msgs/Header header
