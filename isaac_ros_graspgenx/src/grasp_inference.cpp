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

#include "isaac_ros_graspgenx/grasp_inference.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_graspgenx/grasp_processing.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace graspgenx
{
namespace
{

void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

// These private buffers retain exclusive write handles while scratch is in use. They are never
// published; received clouds use read handles. The rosidl CUDA pool owns every device allocation.
class DeviceBuffer
{
public:
  DeviceBuffer(std::size_t bytes, cudaStream_t stream)
  : storage_(cuda_buffer_backend::allocate_buffer(bytes)),
    writer_(cuda_buffer_backend::from_output_buffer(storage_, stream)) {}

  template<typename T = float>
  T * Data() {return reinterpret_cast<T *>(writer_.get_ptr());}

private:
  rosidl::Buffer<uint8_t> storage_;
  cuda_buffer_backend::WriteHandle writer_;
};

// DLPack dtype codes: signed integer = 0, floating point = 2.
tensor_msgs::msg::ExperimentalTensor AllocateTensor(
  std::initializer_list<int64_t> shape, uint8_t code, uint8_t bits)
{
  tensor_msgs::msg::ExperimentalTensor tensor;
  tensor.shape = shape;
  tensor.dtype_code = code;
  tensor.dtype_bits = bits;
  tensor.dtype_lanes = 1;
  size_t bytes = bits / 8;
  for (const auto dimension : shape) {
    bytes *= dimension;
  }
  tensor.data = cuda_buffer_backend::allocate_buffer(bytes);
  return tensor;
}

const tensor_msgs::msg::ExperimentalTensor & CheckedOutput(
  const TensorList & outputs, const std::string & name, std::initializer_list<int64_t> shape)
{
  const auto * tensor = isaac_ros_tensor_msgs::FindTensorByName(outputs, name);
  const size_t bytes = name == "output0" ?
    kModelGrasps * 16U * sizeof(float) : kModelGrasps * sizeof(float);
  if (outputs.names.size() != 2 || outputs.tensors.size() != 2 || tensor == nullptr ||
    tensor->shape != std::vector<int64_t>(shape) || tensor->dtype_code != 2 ||
    tensor->dtype_bits != 32 || tensor->dtype_lanes != 1 ||
    !isaac_ros_tensor_msgs::IsContiguousRowMajor(*tensor) ||
    tensor->byte_offset % sizeof(float) != 0 || tensor->byte_offset > tensor->data.size() ||
    bytes > tensor->data.size() - tensor->byte_offset ||
    tensor->data.get_backend_type() != "cuda")
  {
    throw std::runtime_error("Invalid GraspGenX output tensor: " + name);
  }
  return *tensor;
}

}  // namespace

struct GraspInference::CudaStream
{
  CudaStream() {CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));}
  ~CudaStream() {cudaStreamDestroy(stream);}

  cudaStream_t stream{};
};

GraspInference::~GraspInference() = default;

GraspInference::GraspInference(GraspInferenceOptions options, TensorInference inference)
: options_(std::move(options)), inference_(std::move(inference))
{
  if (!inference_ || !std::isfinite(options_.kappa) || options_.kappa <= 0.0F ||
    !std::isfinite(options_.grid_size_m) || options_.grid_size_m <= 0.0F ||
    !std::isfinite(options_.outlier_threshold_m) || options_.outlier_threshold_m <= 0.0F ||
    options_.outlier_neighbors <= 0 || options_.outlier_neighbors > kMaxNeighbors)
  {
    throw std::invalid_argument("Invalid GraspGenX preprocessing settings or tensor transport");
  }
}

InferenceResult GraspInference::Infer(const InferenceRequest & request) const
{
  // The engine always predicts 200 candidates. These fields only constrain postprocessing, and
  // the confidence threshold must be finite so comparisons in the CUDA decoder are well-defined.
  if (request.num_grasps <= 0 || request.top_k_grasps < -1 ||
    !std::isfinite(request.grasp_confidence_threshold))
  {
    throw std::runtime_error("Invalid grasp count or confidence threshold");
  }
  if (!request.object_cloud) {
    throw std::runtime_error("An object cloud is required");
  }

  // The masked-cloud builder guarantees a packed, little-endian float32 XYZ cloud with valid
  // strides and payload size. Only guard the narrowing conversion used by CUDA kernel metadata.
  const auto & cloud = *request.object_cloud;
  const std::size_t count = static_cast<std::size_t>(cloud.width) * cloud.height;
  if (count < 4 || count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Object cloud must contain between 4 and INT_MAX points");
  }

  // Convert ROS metadata into the small trivially copyable description passed to CUDA kernels.
  PointCloudInputMetadata cloud_metadata{};
  cloud_metadata.count = static_cast<int>(count);
  cloud_metadata.width = cloud.width;
  cloud_metadata.point_step = cloud.point_step;
  cloud_metadata.row_step = cloud.row_step;

  // The builder packs XYZ as three consecutive floats at offsets 0, 4, and 8.
  for (int d = 0; d < 3; ++d) {
    // CUDA consumes a row-major float 3x4 transform rather than the host-side Eigen object.
    for (int column = 0; column < 4; ++column) {
      cloud_metadata.origin_pose_point_cloud[d * 4 +
        column] = request.origin_pose_point_cloud.matrix()(d, column);
    }
    cloud_metadata.xyz_offset[d] = d * sizeof(float);
  }

  // Preserve the zero-copy path: preprocessing expects the masked-cloud builder's CUDA buffer,
  // not a CPU PointCloud2 that would require an implicit upload.
  if (cloud.data.get_backend_type() != "cuda") {
    throw std::runtime_error(
            "Object cloud must retain its CUDA buffer from the masked cloud builder");
  }
  std::lock_guard<std::mutex> lock(inference_mutex_);
  if (!stream_) {
    stream_ = std::make_unique<CudaStream>();
  }
  const auto stream = stream_->stream;
  // The read handle keeps the input allocation alive and orders this stream after its producer.
  auto input = pointcloud_conversions::from_input_point_cloud_msg(cloud, stream);

  // Scratch and tensor storage belong to this request, including while waiting for a reply.
  DeviceBuffer points(count * 3U * sizeof(float), stream);
  DeviceBuffer keep(count * sizeof(int), stream);
  DeviceBuffer indices(count * sizeof(int), stream);
  DeviceBuffer center(3U * sizeof(float), stream);
  DeviceBuffer status(4U * sizeof(int), stream);
  DeviceBuffer tool(16U * sizeof(float), stream);
  DeviceBuffer decoded(kModelGrasps * sizeof(DeviceGrasp), stream);

  // Group request-local scratch pointers into the contract consumed by grasp_processing.cu.
  ProcessingWorkspace workspace{
    points.Data(), keep.Data<int>(), indices.Data<int>(), center.Data(), status.Data<int>()};
  TensorList inputs;
  inputs.header = cloud.header;
  inputs.names = {"coord", "grid_coord", "code", "order", "inverse",
    "sweep_volume", "init_noise", "step_noise", "feat"};
  // Initializer-list copies finalize CUDA buffers before preprocessing can write them.
  inputs.tensors.reserve(inputs.names.size());
  inputs.tensors.push_back(AllocateTensor({kModelPoints, 3}, 2, 32));
  inputs.tensors.push_back(AllocateTensor({kModelPoints, 3}, 0, 32));
  inputs.tensors.push_back(AllocateTensor({kSpatialOrders, kModelPoints}, 0, 64));
  inputs.tensors.push_back(AllocateTensor({kSpatialOrders, kModelPoints}, 0, 64));
  inputs.tensors.push_back(AllocateTensor({kSpatialOrders, kModelPoints}, 0, 64));
  inputs.tensors.push_back(AllocateTensor({1, 12}, 2, 32));
  inputs.tensors.push_back(AllocateTensor({kModelGrasps, 6}, 2, 32));
  inputs.tensors.push_back(AllocateTensor({kDiffusionSteps, kModelGrasps, 6}, 2, 32));
  std::vector<cuda_buffer_backend::WriteHandle> writers;
  for (size_t i = 0; i < inputs.tensors.size(); ++i) {
    writers.push_back(
      cuda_buffer_backend::from_output_buffer(
        inputs.tensors[i].data, stream));
  }
  TensorRtInputTensors tensors{
    reinterpret_cast<float *>(writers[0].get_ptr()),
    reinterpret_cast<int32_t *>(writers[1].get_ptr()),
    reinterpret_cast<int64_t *>(writers[2].get_ptr()),
    reinterpret_cast<int64_t *>(writers[3].get_ptr()),
    reinterpret_cast<int64_t *>(writers[4].get_ptr()),
    reinterpret_cast<float *>(writers[6].get_ptr()),
    reinterpret_cast<float *>(writers[7].get_ptr())};
  const auto & gripper = options_.grippers.at(request.gripper_name);

  // Combine immutable backend preprocessing options with this request's reproducibility seed.
  const PointCloudPreprocessingSettings settings{
    options_.kappa, options_.grid_size_m, options_.outlier_threshold_m,
    options_.outlier_neighbors, static_cast<uint64_t>(request.random_seed)};
  // DecodeGrasps expects the tool offset as a row-major float matrix in device memory.
  const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> tool_matrix =
    request.grasp_pose_tool.matrix().cast<float>();
  // Host mirror of the four device status values:
  // [kept point count, grid error, decoded grasp count, spatial-code bit depth].
  std::array<int, 4> processing_status{};
  std::vector<DeviceGrasp> grasps;

  try {
    // All preparation is ordered on this stream. Release tensor writers before publishing.
    cuda_buffer_backend::to_buffer(
      gripper.data(), sizeof(gripper), writers[5], stream, cudaMemcpyHostToDevice);
    CheckCuda(
      cudaMemcpyAsync(
        tool.Data(), tool_matrix.data(), 16U * sizeof(float), cudaMemcpyHostToDevice,
        stream));
    PrepareModelInputs(
      input.get_ptr(), cloud_metadata, settings, workspace, tensors, stream);
    // Detect synchronous launch/configuration failures before waiting for preprocessing.
    CheckCuda(cudaGetLastError());

    // Only counts/status cross to the CPU. XYZ, spatial orders and noise remain on the device.
    CheckCuda(
      cudaMemcpyAsync(
        processing_status.data(), status.Data<int>(), sizeof(processing_status),
        cudaMemcpyDeviceToHost,
        stream));
    // TensorRT must not start until preprocessing and its status readback have completed.
    CheckCuda(cudaStreamSynchronize(stream));
    if (processing_status[0] < 4 || processing_status[1] != 0) {
      throw std::runtime_error(
              "Insufficient filtered points or unsupported GraspGenX grid extent");
    }

    writers.clear();
    // Alias only after writing, so a shared buffer cannot trigger copy-on-write during preparation.
    inputs.tensors.push_back(inputs.tensors.front());
    const auto outputs = inference_(inputs);
    const auto & poses = CheckedOutput(outputs, "output0", {kModelGrasps, 4, 4});
    const auto & scores = CheckedOutput(outputs, "output1", {kModelGrasps, 1});
    // Read handles wait for TensorRT's output events on its own stream. Retain them until
    // decoding completes; no intermediate tensor is copied through CPU memory.
    auto pose_reader = cuda_buffer_backend::from_input_buffer(poses.data, stream);
    auto score_reader = cuda_buffer_backend::from_input_buffer(scores.data, stream);

    // top_k_grasps is an optional second cap; -1 means num_grasps is the only limit.
    const int limit = request.top_k_grasps < 0 ?
      request.num_grasps : std::min(request.num_grasps, request.top_k_grasps);
    // Validate poses, restore their center/tool transform, filter scores, and rank on the GPU.
    DecodeGrasps(
      reinterpret_cast<const float *>(pose_reader.get_ptr() + poses.byte_offset),
      reinterpret_cast<const float *>(score_reader.get_ptr() + scores.byte_offset),
      center.Data(), tool.Data(),
      request.grasp_confidence_threshold, limit, decoded.Data<DeviceGrasp>(), status.Data<int>(),
      stream);
    CheckCuda(cudaGetLastError());

    // Read status again because DecodeGrasps writes the selected result count to status[2].
    CheckCuda(
      cudaMemcpyAsync(
        processing_status.data(), status.Data<int>(), sizeof(processing_status),
        cudaMemcpyDeviceToHost,
        stream));
    CheckCuda(cudaStreamSynchronize(stream));

    // This is the public action-result boundary: download only the selected poses and scores.
    grasps.resize(processing_status[2]);
    if (!grasps.empty()) {
      CheckCuda(
        cudaMemcpyAsync(
          grasps.data(), decoded.Data<DeviceGrasp>(), grasps.size() * sizeof(DeviceGrasp),
          cudaMemcpyDeviceToHost, stream));
      CheckCuda(cudaStreamSynchronize(stream));
    }
  } catch (...) {
    // Request-local buffers are about to unwind. Drain the stream so no queued GPU operation can
    // access those allocations after their destructors run, then preserve the original error.
    cudaStreamSynchronize(stream);
    throw;
  }

  // Convert the compact device result representation into the public Eigen-based response.
  InferenceResult result;
  for (const auto & grasp : grasps) {
    using Matrix = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.matrix() = Eigen::Map<const Matrix>(grasp.pose).cast<double>();
    result.grasps.push_back({pose, grasp.confidence, GraspSource::kDiffusion});
  }
  // An inference that runs successfully but produces no accepted pose is reported as failure.
  result.success = !result.grasps.empty();
  result.message = result.success ?
    "Generated " + std::to_string(result.grasps.size()) + " diffusion grasp candidates" :
    "No valid GraspGenX diffusion grasps passed the confidence threshold";
  return result;
}

}  // namespace graspgenx
}  // namespace isaac_ros
}  // namespace nvidia
