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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_graspgenx_models_install)

# The asset installer reuses this file to compile the plugin on the target GPU.
# Return before ROS setup so this invocation cannot trigger asset installation.
option(GRASPGENX_BUILD_PLUGIN_ONLY "Build the plugin for model installation" OFF)
if(GRASPGENX_BUILD_PLUGIN_ONLY)
  cmake_minimum_required(VERSION 3.24)
  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
  find_library(TENSORRT_LIBRARY nvinfer REQUIRED)

  # Use the TensorRT release that Isaac ROS is tested with.
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${TENSORRT_INCLUDE_DIR}")
  unset(GRASPGENX_TENSORRT_VERSION_SUPPORTED CACHE)
  check_cxx_source_compiles("
    #include <NvInferVersion.h>
    static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 16,
                  \"GraspGenX requires TensorRT 10.16\");
    int main() { return 0; }
  " GRASPGENX_TENSORRT_VERSION_SUPPORTED)
  cmake_pop_check_state()
  if(NOT GRASPGENX_TENSORRT_VERSION_SUPPORTED)
    message(FATAL_ERROR "The GraspGenX engine builder requires TensorRT 10.16")
  endif()

  set(GRASPGENX_PLUGIN_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_plugin/graspgenx_plugins.cu"
  )
  if(NOT EXISTS "${GRASPGENX_PLUGIN_SOURCE}")
    message(FATAL_ERROR "GraspGenX plugin source not found: ${GRASPGENX_PLUGIN_SOURCE}")
  endif()
  add_library(graspgenx_plugins SHARED "${GRASPGENX_PLUGIN_SOURCE}")
  target_include_directories(graspgenx_plugins PRIVATE "${TENSORRT_INCLUDE_DIR}")
  target_link_libraries(graspgenx_plugins PRIVATE CUDA::cudart CUDA::cublas "${TENSORRT_LIBRARY}")
  set_target_properties(graspgenx_plugins PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  return()
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install_isaac_ros_asset(install_graspgenx_models)
install(PROGRAMS
  asset_scripts/install_graspgenx_models.sh
  asset_scripts/model_manifest_checksums.py
  asset_scripts/export_huggingface_model.py
  asset_scripts/export_huggingface_model.sh
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES CMakeLists.txt DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_model_manifest_checksums test/test_model_manifest_checksums.py)
endif()

ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

install(DIRECTORY vendor/ DESTINATION share/${PROJECT_NAME}/vendor
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
ament_auto_package(INSTALL_TO_SHARE tensorrt_plugin)
