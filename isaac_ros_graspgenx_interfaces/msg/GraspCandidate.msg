# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Grasp pose in meters with a quaternion orientation; header identifies its frame and timestamp.
geometry_msgs/PoseStamped pose

# Discriminator sigmoid score in [0.0, 1.0], used for filtering and ranking;
# higher scores are preferred.
float32 confidence

# Name of the inference method that generated this candidate.
string source
