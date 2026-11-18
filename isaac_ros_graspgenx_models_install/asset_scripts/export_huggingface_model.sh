#!/bin/bash
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

# Run the exporter with its isolated Python dependencies.
set -euo pipefail

# Use the export venv; Isaac ROS activation's interpreter override takes precedence even with -I.
unset PYTHONEXECUTABLE
HF_GGX="${HF_GGX:-${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/isaac_ros_assets/isaac_ros_graspgenx}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${HF_GGX}/venv/bin/python" -I "${SCRIPT_DIR}/export_huggingface_model.py" "$@"
