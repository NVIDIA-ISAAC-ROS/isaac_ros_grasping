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
project(isaac_ros_graspgenx LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Load isaac_ros_common's supported CUDA architectures before enabling CUDA;
# otherwise CMake's compiler default prevents the platform defaults from applying.
enable_language(CUDA)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(GRASPGENX_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/graspgenx")

# Model-specific CUDA processing around a tensor transport supplied by the action server.
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/tensor_inference_client.cpp
  src/generate_batch_input_node.cpp
  src/generate_grasp_node.cpp
  src/impl/transform_utils.cpp
  src/grasp_inference.cpp
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "nvidia::isaac_ros::graspgenx::GenerateGraspNode"
  EXECUTABLE graspgenx_generate_grasp_node
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "nvidia::isaac_ros::graspgenx::GenerateBatchInputNode"
  EXECUTABLE graspgenx_generate_batch_input_node
)

set(
  GRASPGENX_CUDA_ARCHITECTURES
  ""
  CACHE STRING
  "CUDA architectures for GraspGenX plugins; empty uses the CMake/CUDA compiler default"
)

add_library(graspgenx_processing SHARED src/grasp_processing.cu)
target_compile_features(graspgenx_processing PUBLIC cxx_std_17)
target_include_directories(graspgenx_processing PRIVATE include)
target_link_libraries(graspgenx_processing PRIVATE CUDA::cudart)
set_target_properties(graspgenx_processing PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
if(GRASPGENX_CUDA_ARCHITECTURES)
  set_property(
    TARGET graspgenx_processing
    PROPERTY CUDA_ARCHITECTURES "${GRASPGENX_CUDA_ARCHITECTURES}"
  )
endif()
target_link_libraries(${PROJECT_NAME}
  graspgenx_processing
  CUDA::cudart
)

install(
  TARGETS graspgenx_processing
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  PROGRAMS utils/graspgenx_batch_client.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME graspgenx_batch_client
)
install(
  PROGRAMS utils/graspgenx_visualization.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME graspgenx_visualization
)
install(
  DIRECTORY utils
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  file(GLOB_RECURSE GRASPGENX_VENDOR_FILES "${GRASPGENX_VENDOR_DIR}/*")
  set(AMENT_LINT_AUTO_FILE_EXCLUDE ${GRASPGENX_VENDOR_FILES})
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Upstream CPU implementations are reference oracles for the GPU tests only.
  add_library(graspgenx_preprocessing_cpu STATIC
    "${GRASPGENX_VENDOR_DIR}/serialize.cpp"
    "${GRASPGENX_VENDOR_DIR}/graspmoe_preprocess.cpp"
  )
  target_compile_features(graspgenx_preprocessing_cpu PUBLIC cxx_std_17)
  target_include_directories(graspgenx_preprocessing_cpu PUBLIC
    "${GRASPGENX_VENDOR_DIR}"
  )

  foreach(test transform_utils generate_grasp_node tensor_inference_client)
    ament_add_gtest(test_${test} test/test_${test}.cpp)
    target_link_libraries(test_${test} ${PROJECT_NAME})
  endforeach()

  ament_add_gtest(test_grasp_processing test/test_grasp_processing.cpp)
  target_include_directories(test_grasp_processing PRIVATE include)
  target_link_libraries(test_grasp_processing graspgenx_processing graspgenx_preprocessing_cpu CUDA::cudart)

  ament_add_gtest(test_grasp_inference test/test_grasp_inference.cpp)
  target_link_libraries(test_grasp_inference ${PROJECT_NAME})

  foreach(test graspgenx_visualization point_cloud_utils transform_utils)
    ament_add_pytest_test(
      test_${test}_python
      test/test_${test}.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
  endforeach()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(
    test/isaac_ros_graspgenx_pol.py
    TIMEOUT "600"
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE config launch rviz
)
