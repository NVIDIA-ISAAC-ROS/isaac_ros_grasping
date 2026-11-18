// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "isaac_ros_graspgenx/grasp_processing.hpp"
#include "serialize.h"  // NOLINT(build/include_subdir)
#include "graspmoe_preprocess.h"  // NOLINT(build/include_subdir)

namespace nvidia::isaac_ros::graspgenx
{
namespace
{

// Unified allocations keep test fixtures readable; production allocations use rosidl CUDA buffers.
template<typename T>
class Array
{
public:
  explicit Array(std::size_t count)
  {
    if (cudaMallocManaged(reinterpret_cast<void **>(&data), count * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("CUDA test allocation failed");
    }
    std::fill(data, data + count, T{});
  }
  ~Array() {cudaFree(data);}
  T * data{};
};

class GraspProcessingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "A CUDA device is required";
    }
  }
};

TEST_F(GraspProcessingTest, SpatialOrdersMatchUpstreamIncludingDuplicateCells)
{
  Array<float> coord(kModelPoints * 3);
  Array<int32_t> grid(kModelPoints * 3);
  Array<int64_t> code(kModelPoints * kSpatialOrders), order(kModelPoints * kSpatialOrders),
  inverse(kModelPoints * kSpatialOrders);
  Array<int> status(4);
  for (int i = 0; i < kModelPoints; ++i) {
    coord.data[i * 3] = (i % 37) * 0.012F;
    coord.data[i * 3 + 1] = ((i / 37) % 23) * 0.023F;
    coord.data[i * 3 + 2] = (i % 11) * 0.031F;
  }
  const auto reference = ::graspgenx::serialize(coord.data, kModelPoints, 0.01F);
  TensorRtInputTensors tensors{coord.data, grid.data, code.data, order.data, inverse.data, nullptr,
    nullptr};
  SerializePoints(tensors, 0.01F, status.data, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(std::equal(reference.grid_coord.begin(), reference.grid_coord.end(), grid.data));
  EXPECT_TRUE(std::equal(reference.code.begin(), reference.code.end(), code.data));
  EXPECT_TRUE(std::equal(reference.order.begin(), reference.order.end(), order.data));
  EXPECT_TRUE(std::equal(reference.inverse.begin(), reference.inverse.end(), inverse.data));
  std::fill(coord.data, coord.data + kModelPoints * 3, 1.0F);
  SerializePoints(tensors, 0.01F, status.data, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(status.data[3], 0);
  for (int i = 0; i < kModelPoints * kSpatialOrders; ++i) {
    EXPECT_EQ(code.data[i], 0);
    EXPECT_EQ(order.data[i], i % kModelPoints);
  }
}

TEST_F(GraspProcessingTest, StridedCloudFilteringCenteringAndSeedRepeatability)
{
  constexpr int count = 25;
  Array<float> cloud(count * 4), points(count * 3), center(3), coord(kModelPoints * 3);
  Array<int> keep(count), indices(count), status(4), grid(kModelPoints * 3);
  Array<int64_t> code(kModelPoints * kSpatialOrders), order(kModelPoints * kSpatialOrders),
  inverse(kModelPoints * kSpatialOrders);
  Array<float> initial(kModelGrasps * 6), noise(kDiffusionSteps * kModelGrasps * 6);
  for (int i = 0; i < count; ++i) {
    cloud.data[i * 4] = 42.0F;  // Non-XYZ padding field.
    cloud.data[i * 4 + 1] = (i % 5) * 0.001F;
    cloud.data[i * 4 + 2] = (i / 5) * 0.001F;
    cloud.data[i * 4 + 3] = 0.001F;
  }
  cloud.data[1] = 10.0F;  // An outlier.
  PointCloudInputMetadata cloud_metadata{count, count, 16, count * 16, {4, 8, 12},
    {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3}};
  PointCloudPreprocessingSettings settings{3.27F, 0.01F, 0.014F, 20, 7};
  ProcessingWorkspace workspace{points.data, keep.data, indices.data, center.data, status.data};
  TensorRtInputTensors tensors{coord.data, grid.data, code.data, order.data, inverse.data,
    initial.data, noise.data};
  PrepareModelInputs(
    reinterpret_cast<uint8_t *>(cloud.data), cloud_metadata, settings,
    workspace, tensors, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto reference = ::graspgenx::sor_keep_mask_l1(points.data, count, 0.014F, 20);
  EXPECT_TRUE(std::equal(reference.begin(), reference.end(), keep.data));
  EXPECT_EQ(status.data[0], 24);
  EXPECT_EQ(keep.data[0], 0);
  for (int d = 0; d < 3; ++d) {
    double sum = 0;
    for (int i = 0; i < kModelPoints; ++i) {
      sum += coord.data[i * 3 + d];
    }
    EXPECT_NEAR(sum / kModelPoints, 0.0, 1e-3);
    EXPECT_NEAR(center.data[d], d + 1.0, 0.01);
  }
  const std::vector<float> first_coord(coord.data, coord.data + kModelPoints * 3);
  const std::vector<float> first_noise(noise.data, noise.data + kDiffusionSteps * kModelGrasps * 6);
  PrepareModelInputs(
    reinterpret_cast<uint8_t *>(cloud.data), cloud_metadata, settings,
    workspace, tensors, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(std::equal(first_coord.begin(), first_coord.end(), coord.data));
  EXPECT_TRUE(std::equal(first_noise.begin(), first_noise.end(), noise.data));
}

TEST_F(GraspProcessingTest, DecodeRestoresCenterAppliesToolAndRanksStably)
{
  Array<float> matrices(kModelGrasps * 16), scores(kModelGrasps), center(3), tool(16);
  Array<DeviceGrasp> results(kModelGrasps);
  Array<int> status(4);
  for (int d = 0; d < 4; ++d) {
    tool.data[d * 5] = 1.0F;
  }
  center.data[0] = 2.0F;
  tool.data[7] = 3.0F;
  for (int i = 0; i < 4; ++i) {
    for (int d = 0; d < 4; ++d) {
      matrices.data[i * 16 + d * 5] = 1.0F;
    }
    matrices.data[i * 16 + 3] = i;
  }
  scores.data[0] = 0.5F;
  scores.data[1] = 0.9F;
  scores.data[2] = 0.9F;
  scores.data[3] = std::numeric_limits<float>::quiet_NaN();
  DecodeGrasps(
    matrices.data, scores.data, center.data, tool.data,
    0.6F, 2, results.data, status.data, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(status.data[2], 2);
  EXPECT_FLOAT_EQ(results.data[0].pose[3], 3.0F);
  EXPECT_FLOAT_EQ(results.data[1].pose[3], 4.0F);
  EXPECT_FLOAT_EQ(results.data[0].pose[7], 3.0F);
  EXPECT_FLOAT_EQ(results.data[0].confidence, 0.9F);
  DecodeGrasps(
    matrices.data, scores.data, center.data, tool.data,
    0.6F, 0, results.data, status.data, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(status.data[2], 0);
}

}  // namespace
}  // namespace nvidia::isaac_ros::graspgenx
