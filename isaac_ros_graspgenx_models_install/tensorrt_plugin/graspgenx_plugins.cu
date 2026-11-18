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

// Native CUDA implementations of the GraspGenX encoder's TensorRT IPluginV3
// operations: gather0, scatter_reduce, argsort, flash_attn, hash_conv, and
// pool_cluster. All creators use namespace "graspgenx" and version "1".
// Each implementation below documents its input and output tensor layouts.
//
// getGraspgenxCreators() exposes the creators, and AutoRegister registers them
// when this library is loaded, before an engine is built or deserialized.

#include "NvInfer.h"
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

using namespace nvinfer1;

static constexpr int64_t kSentinelBase = (int64_t) 1 << 30; // int32-safe; > any real code

// Lazily-created process-wide cuBLAS handle (stream set per enqueue).
static cublasHandle_t gCublas()
{
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

namespace graspgenx {

// ─── CUDA kernels ────────────────────────────────────────────────────────────

// out[m, :] = src[index[m], :]  (row gather along dim 0). dtype-agnostic:
// copies `rowBytes` bytes per row so it serves float / int32 / int64 src.
__global__ void gather0Kernel(
    char const* src, int64_t const* index, char* out, int32_t M, int32_t rowBytes)
{
    int32_t m = blockIdx.x;
    if (m >= M) return;
    int64_t row = index[m];
    char const* s = src + (int64_t) row * rowBytes;
    char* d = out + (int64_t) m * rowBytes;
    for (int32_t b = threadIdx.x * 4; b < rowBytes; b += blockDim.x * 4)
        *reinterpret_cast<int*>(d + b) = *reinterpret_cast<int const*>(s + b);
}

enum class Reduce { kAMAX, kSUM, kAMIN };

// Fixed-capacity scatter-reduce over dim 0. index[m,c] is in [0,N).
// The reduction includes caller-provided initial values already written into
// `out` (-inf for amax, 0 for sum, +inf for amin).
__global__ void scatterReduceKernel(
    int64_t const* index, float const* src, float* out, int32_t M, int32_t cols,
    Reduce op)
{
    int32_t m = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M) return;
    for (int32_t c = threadIdx.x; c < cols; c += blockDim.x)
    {
        int64_t dst = index[m * cols + c];
        float v = src[m * cols + c];
        float* addr = &out[dst * cols + c];
        if (op == Reduce::kSUM)
            atomicAdd(addr, v);
        else if (op == Reduce::kAMAX)
            atomicMax((int*) addr, __float_as_int(v)); // valid for the ranges here (finite, mixed sign handled below)
        else
            atomicMin((int*) addr, __float_as_int(v));
    }
}

// atomicMax/Min on float via int bit-tricks is only monotonic for
// non-negative floats; the encoder's amax runs on arbitrary-sign features, so
// use a CAS loop for correctness.
__global__ void scatterReduceCASKernel(
    int64_t const* index, float const* src, float* out, int32_t M, int32_t cols,
    Reduce op)
{
    int32_t m = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M) return;
    for (int32_t c = threadIdx.x; c < cols; c += blockDim.x)
    {
        int64_t dst = index[m * cols + c];
        float v = src[m * cols + c];
        int* addr = (int*) &out[dst * cols + c];
        int old = *addr, assumed;
        do {
            assumed = old;
            float cur = __int_as_float(assumed);
            float nv = (op == Reduce::kAMAX) ? fmaxf(cur, v) : fminf(cur, v);
            old = atomicCAS(addr, assumed, __float_as_int(nv));
        } while (assumed != old);
    }
}

static int64_t volume(Dims const& d, int32_t from = 0)
{
    int64_t v = 1;
    for (int32_t i = from; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

// ─── gather0 plugin ──────────────────────────────────────────────────────────

class Gather0Plugin : public IPluginV3, public IPluginV3OneCore,
                      public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    Gather0Plugin() = default;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new Gather0Plugin(*this); }

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override { return "gather0"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const* in, int32_t) const noexcept override
    {
        out[0] = in[0]; // same dtype as src
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t,
        DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override
    {
        // out shape = [len(index), src.shape[1:]]
        outputs[0].nbDims = inputs[0].nbDims;
        outputs[0].d[0] = inputs[1].d[0];
        for (int32_t i = 1; i < inputs[0].nbDims; ++i) outputs[0].d[i] = inputs[0].d[i];
        return 0;
    }
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    {
        auto const& d = io[pos].desc;
        if (d.format != TensorFormat::kLINEAR) return false;
        if (pos == 1) return d.type == DataType::kINT64; // index
        // src (0) and out (2): any of float/int32/int64, and out must match src.
        auto t = d.type;
        bool okType = (t == DataType::kFLOAT || t == DataType::kINT32 || t == DataType::kINT64);
        if (pos == 2) return okType && d.type == io[0].desc.type;
        return okType;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 0; }

    // IPluginV3OneRuntime
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int32_t M = inDesc[1].dims.d[0];
        int32_t cols = (int32_t) volume(inDesc[0].dims, 1);
        int esz = inDesc[0].type == DataType::kINT64 ? 8 : 4; // int64 = 8B, float/int32 = 4B
        gather0Kernel<<<M, 256, 0, stream>>>(
            (char const*) inputs[0], (int64_t const*) inputs[1], (char*) outputs[0], M, cols * esz);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        mFC.nbFields = 0; mFC.fields = nullptr; return &mFC;
    }

private:
    PluginFieldCollection mFC{};
};

// ─── scatter_reduce plugin ───────────────────────────────────────────────────
// inputs: dst[N,C] (init values), index[M,C] int64, src[M,C]. attr: reduce.
// output: reduced[N,C].

class ScatterReducePlugin : public IPluginV3, public IPluginV3OneCore,
                            public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    explicit ScatterReducePlugin(std::string reduce = "amax") : mReduce(std::move(reduce)) {}

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new ScatterReducePlugin(*this); }

    AsciiChar const* getPluginName() const noexcept override { return "scatter_reduce"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }

    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const* in, int32_t) const noexcept override
    {
        out[0] = in[0]; return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t,
        DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override
    {
        outputs[0] = inputs[0]; return 0; // same shape as dst
    }
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    {
        auto const& d = io[pos].desc;
        if (pos == 1) return d.type == DataType::kINT64 && d.format == TensorFormat::kLINEAR; // index
        return d.type == DataType::kFLOAT && d.format == TensorFormat::kLINEAR;               // dst/src/out
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const* outDesc,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int32_t N = outDesc[0].dims.d[0];
        int32_t C = (int32_t) volume(outDesc[0].dims, 1);
        int32_t M = inDesc[1].dims.d[0];
        // Initialize output with dst (the init values).
        cudaMemcpyAsync(outputs[0], inputs[0], (size_t) N * C * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        Reduce op = mReduce == "sum" ? Reduce::kSUM : (mReduce == "amin" ? Reduce::kAMIN : Reduce::kAMAX);
        dim3 block(32, 8);
        dim3 grid((M + block.y - 1) / block.y);
        if (op == Reduce::kSUM)
            scatterReduceKernel<<<grid, block, 0, stream>>>(
                (int64_t const*) inputs[1], (float const*) inputs[2], (float*) outputs[0], M, C, op);
        else
            scatterReduceCASKernel<<<grid, block, 0, stream>>>(
                (int64_t const*) inputs[1], (float const*) inputs[2], (float*) outputs[0], M, C, op);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        mReduceField = mReduce;
        mFields[0] = PluginField("reduce", mReduceField.c_str(), PluginFieldType::kCHAR, (int32_t) mReduceField.size());
        mFC.nbFields = 1; mFC.fields = mFields; return &mFC;
    }

private:
    std::string mReduce;
    std::string mReduceField;
    PluginField mFields[1];
    PluginFieldCollection mFC{};
};

// ─── argsort plugin (1-D int64, stable ascending) ───────────────────────────
// out[i] = index that sorts x ascending. Uses a stable sort by key so ties
// break by original position. Workspace holds a scratch copy of the keys
// because the sort modifies them in place.

class ArgsortPlugin : public IPluginV3, public IPluginV3OneCore,
                      public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    ArgsortPlugin() = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new ArgsortPlugin(*this); }
    AsciiChar const* getPluginName() const noexcept override { return "argsort"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override
    {
        out[0] = DataType::kINT64; return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t,
        DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override
    {
        outputs[0] = inputs[0]; return 0;
    }
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    {
        auto const& d = io[pos].desc;
        return d.type == DataType::kINT64 && d.format == TensorFormat::kLINEAR;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return (size_t) in[0].max.d[0] * sizeof(int64_t); // scratch keys
    }
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        int32_t N = inDesc[0].dims.d[0];
        int64_t* keys = (int64_t*) workspace;
        int64_t* idx = (int64_t*) outputs[0];
        cudaMemcpyAsync(keys, inputs[0], (size_t) N * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<int64_t> kp(keys), ip(idx);
        thrust::sequence(policy, ip, ip + N);
        thrust::stable_sort_by_key(policy, kp, kp + N, ip);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
private:
    PluginFieldCollection mFC{};
};

class ArgsortCreator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "argsort"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new ArgsortPlugin();
    }
private:
    PluginFieldCollection mFC{};
};

// ─── flash_attn plugin ───────────────────────────────────────────────────────
// Scaled dot-product attention with an additive key-bias mask. Inputs:
//   q,k,v [P,H,K,D], bias[P,1,1,K]. Output [P,H,K,D]. scale = 1/sqrt(D).
// Attention scores and output use batched matrix multiplication in fp32.
// Softmax uses one CUDA block per (patch, head, query), over the K keys.

// Row-wise softmax over the K keys, with additive per-key bias (broadcast
// over query rows) and scale. scores is [BH, K, K] row-major (query x key);
// bias is [P, K], patch p = bh / H.
__global__ void softmaxBiasKernel(float* scores, float const* bias, float scale,
    int32_t BH, int32_t H, int32_t K)
{
    int row = blockIdx.x;   // 0..BH*K-1 : one query row
    if (row >= BH * K) return;
    int bh = row / K;
    int p = bh / H;
    float* s = scores + (int64_t) row * K;
    float const* bp = bias + (int64_t) p * K;
    int t = threadIdx.x, nt = blockDim.x;
    __shared__ float red[256];
    float m = -1e30f;
    for (int j = t; j < K; j += nt) { float x = s[j] * scale + bp[j]; s[j] = x; m = fmaxf(m, x); }
    red[t] = m; __syncthreads();
    for (int q = nt/2; q>0; q>>=1){ if(t<q) red[t]=fmaxf(red[t],red[t+q]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float ps = 0.f;
    for (int j = t; j < K; j += nt) { float e = __expf(s[j]-m); s[j]=e; ps+=e; }
    red[t] = ps; __syncthreads();
    for (int q = nt/2; q>0; q>>=1){ if(t<q) red[t]+=red[t+q]; __syncthreads(); }
    float inv = 1.f / (red[0] + 1e-20f);
    for (int j = t; j < K; j += nt) s[j] *= inv;
}

class FlashAttnPlugin : public IPluginV3, public IPluginV3OneCore,
                        public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    FlashAttnPlugin() = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override
    {
        if (t == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new FlashAttnPlugin(*this); }
    AsciiChar const* getPluginName() const noexcept override { return "flash_attn"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* o, int32_t, DataType const*, int32_t) const noexcept override
    { o[0] = DataType::kFLOAT; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t,
        DimsExprs* out, int32_t, IExprBuilder&) noexcept override
    { out[0] = in[0]; return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    { return io[pos].desc.type == DataType::kFLOAT && io[pos].desc.format == TensorFormat::kLINEAR; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        auto const& q = in[0].max; // [P,H,K,D]
        return (size_t) q.d[0] * q.d[1] * q.d[2] * q.d[2] * sizeof(float); // scores [BH,K,K]
    }
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void* ws, cudaStream_t stream) noexcept override
    {
        auto const& qd = inDesc[0].dims; // [P,H,K,D]
        int32_t P = qd.d[0], H = qd.d[1], K = qd.d[2], D = qd.d[3];
        int32_t BH = P * H;
        float const* q = (float const*) inputs[0];
        float const* kk = (float const*) inputs[1];
        float const* v = (float const*) inputs[2];
        float const* bias = (float const*) inputs[3];
        float* out = (float*) outputs[0];
        float* scores = (float*) ws; // [BH, K, K] row-major (query x key)
        cublasHandle_t h = gCublas();
        cublasSetStream(h, stream);
        float one = 1.f, zero = 0.f;
        // scores = Q @ K^T per batch -> row-major [K(query),K(key)] via the
        // col-major swap trick: C[Kk,Kq] = K^T ... = scores row-major.
        cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, K, K, D,
            &one, kk, D, (long long) K * D, q, D, (long long) K * D,
            &zero, scores, K, (long long) K * K, BH);
        // softmax over keys with scale + additive bias
        float scale = 1.0f / sqrtf((float) D);
        int blk = K < 256 ? ((K + 31) / 32) * 32 : 256; if (blk < 32) blk = 32;
        softmaxBiasKernel<<<BH * K, blk, 0, stream>>>(scores, bias, scale, BH, H, K);
        // out = attn @ V per batch -> row-major [K(query),D]
        cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, K, K,
            &one, v, D, (long long) K * D, scores, K, (long long) K * K,
            &zero, out, D, (long long) K * D, BH);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
private:
    PluginFieldCollection mFC{};
};

class FlashAttnCreator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "flash_attn"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override { return new FlashAttnPlugin(); }
private:
    PluginFieldCollection mFC{};
};

// ─── hash_conv plugin ────────────────────────────────────────────────────────
// Submanifold sparse convolution: hash voxel coordinates, locate neighbors
// in sorted keys, and multiply gathered features by the kernel weights.
// Inputs: feat[N,Cin], grid_coord[N,3] int32, batch[N] int64,
// weight[K3,Cin,Cout], bias[Cout], offsets[K3,3] int64. Output: [N,Cout].

__device__ __forceinline__ int64_t hashKey(int64_t b, int64_t x, int64_t y, int64_t z)
{
    // Prime multipliers for the spatial coordinate hash.
    return b * 334214467LL + x * 73856093LL + y * 19349669LL + z * 83492791LL;
}

__global__ void hashKeysKernel(int64_t const* batch, int32_t const* grid, int64_t* keys, int32_t N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    keys[n] = hashKey(batch[n], grid[n * 3], grid[n * 3 + 1], grid[n * 3 + 2]);
}

// Build neighbor_feat[N, K3*Cin]: for each (n,k) find the voxel at
// grid[n]+offsets[k] via binary search in sorted keys; gather feat or zero.
__global__ void neighborFeatKernel(
    float const* feat, int32_t const* grid, int64_t const* batch, int64_t const* offsets,
    int64_t const* sorted_keys, int64_t const* sort_idx,
    float* nf, int32_t N, int32_t K3, int32_t Cin)
{
    int n = blockIdx.x;
    int k = blockIdx.y;
    if (n >= N || k >= K3) return;
    int64_t nk = hashKey(batch[n],
        grid[n * 3] + offsets[k * 3], grid[n * 3 + 1] + offsets[k * 3 + 1],
        grid[n * 3 + 2] + offsets[k * 3 + 2]);
    // lower_bound
    int lo = 0, hi = N;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (sorted_keys[mid] < nk) lo = mid + 1; else hi = mid; }
    bool found = (lo < N) && (sorted_keys[lo] == nk);
    int64_t idx = found ? sort_idx[lo] : 0;
    float* dst = nf + ((int64_t) n * K3 + k) * Cin;
    if (found)
        for (int i = threadIdx.x; i < Cin; i += blockDim.x) dst[i] = feat[idx * Cin + i];
    else
        for (int i = threadIdx.x; i < Cin; i += blockDim.x) dst[i] = 0.0f;
}

__global__ void addBiasKernel(float* out, float const* bias, int32_t N, int32_t Cout)
{
    int n = blockIdx.x; int o = threadIdx.x;
    for (; o < Cout; o += blockDim.x) out[n * Cout + o] += bias[o];
}

class HashConvPlugin : public IPluginV3, public IPluginV3OneCore,
                       public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    HashConvPlugin() = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override
    {
        if (t == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new HashConvPlugin(*this); }
    AsciiChar const* getPluginName() const noexcept override { return "hash_conv"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* o, int32_t, DataType const*, int32_t) const noexcept override
    { o[0] = DataType::kFLOAT; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t,
        DimsExprs* out, int32_t, IExprBuilder&) noexcept override
    {
        out[0].nbDims = 2;
        out[0].d[0] = in[0].d[0];    // N
        out[0].d[1] = in[3].d[2];    // Cout (weight[K3,Cin,Cout])
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    {
        auto const& d = io[pos].desc;
        if (d.format != TensorFormat::kLINEAR) return false;
        if (pos == 1) return d.type == DataType::kINT32;  // grid_coord
        if (pos == 2 || pos == 5) return d.type == DataType::kINT64; // batch, offsets
        return d.type == DataType::kFLOAT; // feat/weight/bias/out
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        int64_t N = in[0].max.d[0], Cin = in[0].max.d[1], K3 = in[5].max.d[0];
        return (size_t)(N * K3 * Cin) * sizeof(float)   // neighbor_feat
             + (size_t)(2 * N) * sizeof(int64_t);       // keys + sort_idx (sorted keys reuse)
    }
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const* outDesc,
        void const* const* inputs, void* const* outputs, void* ws, cudaStream_t stream) noexcept override
    {
        int32_t N = inDesc[0].dims.d[0];
        int32_t Cin = inDesc[0].dims.d[1];
        int32_t K3 = inDesc[5].dims.d[0];
        int32_t Cout = outDesc[0].dims.d[1];
        float const* feat = (float const*) inputs[0];
        int32_t const* grid = (int32_t const*) inputs[1];
        int64_t const* batch = (int64_t const*) inputs[2];
        float const* weight = (float const*) inputs[3];
        float const* bias = (float const*) inputs[4];
        int64_t const* offsets = (int64_t const*) inputs[5];
        float* out = (float*) outputs[0];

        // int64 arrays first (workspace base is aligned) so they stay 8-byte
        // aligned; the float neighbor_feat follows.
        int64_t* keys = (int64_t*) ws;
        int64_t* sort_idx = keys + N;
        float* nf = (float*) (sort_idx + N);

        auto pol = thrust::cuda::par.on(stream);
        int gx = (N + 255) / 256;
        hashKeysKernel<<<gx, 256, 0, stream>>>(batch, grid, keys, N);
        thrust::device_ptr<int64_t> kp(keys), ip(sort_idx);
        thrust::sequence(pol, ip, ip + N);
        thrust::stable_sort_by_key(pol, kp, kp + N, ip); // keys now sorted

        dim3 g2(N, K3);
        neighborFeatKernel<<<g2, 128, 0, stream>>>(feat, grid, batch, offsets, keys, sort_idx, nf, N, K3, Cin);

        // out[N,Cout] = nf[N, K3*Cin] @ weight[K3*Cin, Cout]  (see notes above)
        int M = K3 * Cin;
        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t h = gCublas();
        cublasSetStream(h, stream);
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Cout, N, M,
            &alpha, weight, Cout, nf, M, &beta, out, Cout);
        addBiasKernel<<<N, 256, 0, stream>>>(out, bias, N, Cout);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
private:
    PluginFieldCollection mFC{};
};

class HashConvCreator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "hash_conv"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override { return new HashConvPlugin(); }
private:
    PluginFieldCollection mFC{};
};

// ─── pool_cluster plugin ─────────────────────────────────────────────────────
// Group points with equal sorted spatial codes into clusters. Inputs:
//   c0[N] int64 (valid codes, invalid = sentinel), code_shifted[O,N] int64,
//   n[] int64 (live count). Output: one packed int64 tensor, length
//   3N + 1 + 3*O*N = [sidx(N), cluster(N), head_idx(N), n_new(1),
//                     new_code(O*N), new_order(O*N), new_inverse(O*N)].

__global__ void flagsKernel(int64_t const* sorted_c, int64_t const* nPtr, int64_t* flags, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int64_t n = *nPtr;
    int64_t prev = (i > 0) ? sorted_c[i - 1] : -1;
    flags[i] = ((sorted_c[i] != prev) && (i < n)) ? 1 : 0;
}

// After csum = inclusive_scan(flags): fill cluster[], head_pos[], and copy n_new.
__global__ void clusterKernel(int64_t const* flags, int64_t const* csum, int64_t const* nPtr,
    int64_t* cluster, int64_t* head_pos, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int64_t n = *nPtr;
    bool valid = i < n;
    int64_t c = csum[i] - 1; if (c < 0) c = 0;
    cluster[i] = valid ? c : (int64_t)(N - 1);
    if (flags[i]) head_pos[csum[i] - 1] = i; // first (and boundary) index of this cluster
}

__global__ void headIdxKernel(int64_t const* sidx, int64_t const* head_pos, int64_t* head_idx, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int64_t hp = head_pos[i]; if (hp > N - 1) hp = N - 1; if (hp < 0) hp = 0;
    head_idx[i] = sidx[hp];
}

__global__ void newCodeKernel(int64_t const* code, int64_t const* head_idx, int64_t const* nNewPtr,
    int64_t* new_code, int32_t O, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y;
    if (i >= N || o >= O) return;
    int64_t n_new = *nNewPtr;
    new_code[o * N + i] = (i < n_new) ? code[o * N + head_idx[i]] : (kSentinelBase + i);
}

__global__ void inverseKernel(int64_t const* new_order, int64_t* new_inverse, int32_t O, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y;
    if (i >= N || o >= O) return;
    new_inverse[o * N + new_order[o * N + i]] = i;
}

class PoolClusterPlugin : public IPluginV3, public IPluginV3OneCore,
                          public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    PoolClusterPlugin() = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override
    {
        if (t == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    IPluginV3* clone() noexcept override { return new PoolClusterPlugin(*this); }
    AsciiChar const* getPluginName() const noexcept override { return "pool_cluster"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* o, int32_t, DataType const*, int32_t) const noexcept override
    { o[0] = DataType::kINT64; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t,
        DimsExprs* out, int32_t, IExprBuilder& eb) noexcept override
    {
        auto N = in[0].d[0];                 // c0[N]
        auto O = in[1].d[0];                 // code[O,N]
        auto three = eb.constant(3);
        auto one = eb.constant(1);
        // 3*N + 1 + 3*O*N
        auto t1 = eb.operation(DimensionOperation::kPROD, *three, *N);
        auto ON = eb.operation(DimensionOperation::kPROD, *O, *N);
        auto t2 = eb.operation(DimensionOperation::kPROD, *three, *ON);
        auto s1 = eb.operation(DimensionOperation::kSUM, *t1, *one);
        auto tot = eb.operation(DimensionOperation::kSUM, *s1, *t2);
        out[0].nbDims = 1; out[0].d[0] = tot;
        return 0;
    }
    bool supportsFormatCombination(int32_t, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override
    { return io[0].desc.type == DataType::kINT64 && io[0].desc.format == TensorFormat::kLINEAR; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        int64_t N = in[0].max.d[0];
        // sorted_c, sidx, flags, csum, head_pos, plus per-order sort scratch (keys)
        return (size_t)(6 * N) * sizeof(int64_t);
    }
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void* ws, cudaStream_t stream) noexcept override
    {
        int32_t N = inDesc[0].dims.d[0];
        int32_t O = inDesc[1].dims.d[0];
        int64_t const* c0 = (int64_t const*) inputs[0];
        int64_t const* code = (int64_t const*) inputs[1];
        int64_t const* nPtr = (int64_t const*) inputs[2];
        int64_t* out = (int64_t*) outputs[0];
        // Output layout offsets.
        int64_t* sidx = out;                 // [N]
        int64_t* cluster = out + N;          // [N]
        int64_t* head_idx = out + 2 * N;     // [N]
        int64_t* n_new = out + 3 * N;        // [1]
        int64_t* new_code = out + 3 * N + 1;             // [O*N]
        int64_t* new_order = new_code + (int64_t) O * N; // [O*N]
        int64_t* new_inverse = new_order + (int64_t) O * N; // [O*N]

        int64_t* w = (int64_t*) ws;
        int64_t* sorted_c = w;               // [N]
        int64_t* flags = w + N;              // [N]
        int64_t* csum = w + 2 * N;           // [N]
        int64_t* head_pos = w + 3 * N;       // [N]
        int64_t* scratch = w + 4 * N;        // [N] per-order sort keys

        auto pol = thrust::cuda::par.on(stream);
        int gx = (N + 255) / 256;

        // argsort(c0) -> sidx (stable), sorted_c = c0[sidx]
        cudaMemcpyAsync(sorted_c, c0, (size_t) N * 8, cudaMemcpyDeviceToDevice, stream);
        thrust::device_ptr<int64_t> scp(sorted_c), sip(sidx);
        thrust::sequence(pol, sip, sip + N);
        thrust::stable_sort_by_key(pol, scp, scp + N, sip);

        // flags, cumsum, cluster, head_pos
        cudaMemsetAsync(head_pos, 0, (size_t) N * 8, stream); // init; overwritten for valid clusters
        flagsKernel<<<gx, 256, 0, stream>>>(sorted_c, nPtr, flags, N);
        thrust::device_ptr<int64_t> fp(flags), cp(csum);
        thrust::inclusive_scan(pol, fp, fp + N, cp);
        // n_new = csum[N-1]
        cudaMemcpyAsync(n_new, csum + (N - 1), 8, cudaMemcpyDeviceToDevice, stream);
        clusterKernel<<<gx, 256, 0, stream>>>(flags, csum, nPtr, cluster, head_pos, N);
        headIdxKernel<<<gx, 256, 0, stream>>>(sidx, head_pos, head_idx, N);

        // new_code[O,N]
        dim3 g2(gx, O);
        newCodeKernel<<<g2, 256, 0, stream>>>(code, head_idx, n_new, new_code, O, N);
        // new_order[o] = argsort(new_code[o]); new_inverse
        for (int o = 0; o < O; ++o)
        {
            cudaMemcpyAsync(scratch, new_code + (int64_t) o * N, (size_t) N * 8, cudaMemcpyDeviceToDevice, stream);
            thrust::device_ptr<int64_t> kp(scratch), op(new_order + (int64_t) o * N);
            thrust::sequence(pol, op, op + N);
            thrust::stable_sort_by_key(pol, kp, kp + N, op);
        }
        inverseKernel<<<g2, 256, 0, stream>>>(new_order, new_inverse, O, N);
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
private:
    PluginFieldCollection mFC{};
};

class PoolClusterCreator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "pool_cluster"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override { return new PoolClusterPlugin(); }
private:
    PluginFieldCollection mFC{};
};

// ─── Creators ────────────────────────────────────────────────────────────────

class Gather0Creator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "gather0"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override { mFC.nbFields = 0; mFC.fields = nullptr; return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new Gather0Plugin();
    }
private:
    PluginFieldCollection mFC{};
};

class ScatterReduceCreator : public IPluginCreatorV3One
{
public:
    AsciiChar const* getPluginName() const noexcept override { return "scatter_reduce"; }
    AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
    AsciiChar const* getPluginNamespace() const noexcept override { return "graspgenx"; }
    PluginFieldCollection const* getFieldNames() noexcept override
    {
        mFields[0] = PluginField("reduce", nullptr, PluginFieldType::kCHAR, 0);
        mFC.nbFields = 1; mFC.fields = mFields; return &mFC;
    }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override
    {
        std::string reduce = "amax";
        if (fc)
            for (int32_t i = 0; i < fc->nbFields; ++i)
                if (std::strcmp(fc->fields[i].name, "reduce") == 0 && fc->fields[i].data)
                    reduce.assign((char const*) fc->fields[i].data, fc->fields[i].length);
        return new ScatterReducePlugin(reduce);
    }
private:
    PluginField mFields[1];
    PluginFieldCollection mFC{};
};

} // namespace graspgenx

// Expose all plugin creators to a host loading this library.
extern "C" nvinfer1::IPluginCreatorV3One* const* getGraspgenxCreators(int32_t* count)
{
    static graspgenx::Gather0Creator gather0;
    static graspgenx::ScatterReduceCreator scatter;
    static graspgenx::ArgsortCreator argsort;
    static graspgenx::PoolClusterCreator poolcluster;
    static graspgenx::HashConvCreator hashconv;
    static graspgenx::FlashAttnCreator flashattn;
    static nvinfer1::IPluginCreatorV3One* creators[] = {&gather0, &scatter, &argsort, &poolcluster, &hashconv, &flashattn};
    *count = 6;
    return creators;
}

// Register all creators with the shared TensorRT registry when this library
// is loaded, making them available for engine construction and deserialization.
namespace {
struct AutoRegister
{
    AutoRegister()
    {
        int32_t n; auto c = getGraspgenxCreators(&n);
        nvinfer1::IPluginRegistry* reg = ::getPluginRegistry();
        if (reg)
            for (int32_t i = 0; i < n; ++i) reg->registerCreator(*c[i], "graspgenx");
    }
} gAutoRegister;
} // namespace
