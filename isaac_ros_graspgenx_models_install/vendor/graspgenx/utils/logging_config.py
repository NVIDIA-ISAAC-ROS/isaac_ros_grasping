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

import logging
import sys

# Global flag to track if logging is initialized
_logging_initialized = False


def setup_logging():
    """
    Set up basic logging configuration for all scripts.
    Uses INFO level and outputs to console.
    """
    global _logging_initialized

    if _logging_initialized:
        return

    # Set the level explicitly. `logging.basicConfig` is a no-op when the
    # root logger already has handlers (e.g. under pytest), which would
    # leave the level at WARNING.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in root_logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(handler)

    _logging_initialized = True


def get_logger(name):
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module (usually __name__)

    Returns:
        A logger instance configured for the module
    """
    # Ensure logging is initialized before getting a logger
    setup_logging()
    return logging.getLogger(name)


# Initialize logging when this module is imported
setup_logging()
