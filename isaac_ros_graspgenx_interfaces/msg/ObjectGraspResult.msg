# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Identifier copied from the corresponding input ObjectCloudReference.
uint64 object_id

# Ranked grasp candidates; each stamped pose carries its own frame and timestamp.
isaac_ros_graspgenx_interfaces/GraspCandidate[] candidates

# True when grasp generation completed successfully for this object.
bool success
