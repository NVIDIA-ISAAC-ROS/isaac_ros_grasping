// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <limits>
#include <stdexcept>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "gtest/gtest.h"
#include "isaac_ros_graspgenx/grasp_inference.hpp"

namespace nvidia::isaac_ros::graspgenx
{
TEST(GraspInference, RequiresTensorTransport)
{
  EXPECT_THROW(GraspInference(GraspInferenceOptions{}, {}), std::invalid_argument);
}

TEST(GraspInference, RejectsInvalidPreprocessingBeforeAllocatingGpuMemory)
{
  GraspInferenceOptions options;
  options.kappa = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(
    GraspInference(options, [](const TensorList &) {return TensorList{};}),
    std::invalid_argument);
}

TEST(GraspInference, MissingCloudDoesNotSubmitInference)
{
  bool submitted = false;
  GraspInference inference(GraspInferenceOptions{}, [&submitted](const TensorList &) {
      submitted = true;
      return TensorList{};
    });
  EXPECT_THROW(inference.Infer(InferenceRequest{}), std::runtime_error);
  EXPECT_FALSE(submitted);
}
TEST(GraspInference, PreparesCudaTensorsBeforeSubmittingTransport)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "A CUDA device is required";
  }
  auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  cloud->width = 100;
  cloud->height = 1;
  cloud->point_step = 3 * sizeof(float);
  cloud->row_step = cloud->width * cloud->point_step;
  std::array<float, 300> points{};
  for (size_t i = 0; i < cloud->width; ++i) {
    points[3 * i] = static_cast<float>(i % 10) * 0.001F;
    points[3 * i + 1] = static_cast<float>(i / 10) * 0.001F;
    points[3 * i + 2] = 0.5F;
  }
  cloud->data = cuda_buffer_backend::allocate_buffer(sizeof(points));
  {
    auto writer = cuda_buffer_backend::from_output_buffer(cloud->data, nullptr);
    cuda_buffer_backend::to_buffer(
      points.data(), sizeof(points), writer, nullptr, cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  }
  GraspInferenceOptions options;
  options.grippers.emplace("test", std::array<float, 12>{});
  bool submitted = false;
  GraspInference inference(options, [&submitted](const TensorList & inputs) -> TensorList {
      submitted = true;
      EXPECT_EQ(inputs.names.size(), 9U);
      EXPECT_EQ(inputs.tensors.size(), inputs.names.size());
      for (const auto & tensor : inputs.tensors) {
        EXPECT_EQ(tensor.data.get_backend_type(), "cuda");
        EXPECT_NO_THROW(cuda_buffer_backend::from_input_buffer(tensor.data, nullptr));
      }
      throw std::runtime_error("Transport reached after CUDA preprocessing");
    });
  InferenceRequest request;
  request.object_cloud = cloud;
  request.gripper_name = "test";
  EXPECT_THROW(inference.Infer(request), std::runtime_error);
  EXPECT_TRUE(submitted);
}
}  // namespace nvidia::isaac_ros::graspgenx
