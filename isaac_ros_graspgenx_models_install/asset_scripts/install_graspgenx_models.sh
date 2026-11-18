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

# Build a TensorRT engine on the target GPU from the Hugging Face ONNX export.
set -euo pipefail

if [[ -z ${ISAAC_ROS_WS:-} && -n ${ISAAC_ROS_ASSET_MODEL_PATH:-} ]]; then
  ISAAC_ROS_WS="$(readlink -f "$(dirname "${ISAAC_ROS_ASSET_MODEL_PATH}")/../../..")"
fi
ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
# Prepare the Python environment before running export_huggingface_model.py.
# Usage: bash install_graspgenx_models.sh --install-export-deps
if [[ ${1:-} == --install-export-deps ]]; then
  # Prevent Isaac ROS activation's interpreter override from sending pip to the shared venv.
  unset PYTHONEXECUTABLE
  HF_GGX="${HF_GGX:-${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_graspgenx}"
  CONSTRAINTS=/usr/share/isaac-ros-cli/pip_shim_constraints.txt
  if [[ ${ISAAC_ROS_PLATFORM:-} == arm64-fastos || -f /etc/fastos-release ]]; then
    CONSTRAINTS=/usr/share/isaac-ros-cli/pip_shim_constraints_fastos.txt
  fi
  # Use Isaac ROS's platform-specific CUDA wheels and export package versions.
  # Keep the NumPy pin so dependency resolution cannot select the incompatible NumPy 2.x ABI.
  EXPORT_CONSTRAINTS="$(sed -n -E '/^(numpy|torch|torchvision|onnx|onnxscript)==/p' "${CONSTRAINTS}")"
  mapfile -t EXPORT_PACKAGES <<<"${EXPORT_CONSTRAINTS}"
  CUDA_WHEEL="$(sed -n 's/^torch==.*+//p' <<<"${EXPORT_CONSTRAINTS}")"
  python3 -m venv "${HF_GGX}/venv"
  "${HF_GGX}/venv/bin/python" -I -m pip install \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}" \
    --constraint <(printf '%s\n' "${EXPORT_CONSTRAINTS}") \
    "${EXPORT_PACKAGES[@]}" \
    omegaconf==2.3.0 addict==2.4.0 timm==1.0.15 diffusers==0.11.1 \
    huggingface-hub==0.25.2
  exit 0
fi

ASSET_NAME="graspgenx"
MODELS_DIR="${ISAAC_ROS_GRASPGENX_ENGINE_DIR:-${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}}"
MODEL_DIR="${ISAAC_ROS_GRASPGENX_MODEL_DIR:-${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_graspgenx/model}"
# Used by the sourced isaac_ros_asset_eula.sh helper.
# shellcheck disable=SC2034
ASSET_DIR="${MODELS_DIR}"
# Used by the sourced isaac_ros_asset_eula.sh helper.
# shellcheck disable=SC2034
ASSET_INSTALL_PATHS="${MODELS_DIR}/graspgenx_everything_fp16.engine ${MODELS_DIR}/libgraspgenx_plugins.so ${MODELS_DIR}/engine_build.json"
# Used by the sourced isaac_ros_asset_eula.sh helper.
# shellcheck disable=SC2034
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/isaac_ros_grasping_assets"
FORCE_BUILD=false
for argument in "$@"; do
  if [[ ${argument} == --no-cache ]]; then FORCE_BUILD=true; fi
done

# Always reach the builder to check engine compatibility with the visible GPU.
# Engine caching is handled below rather than by the EULA helper.
set +u
# shellcheck source=/dev/null
source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}" "$@" --no-cache
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${MODELS_DIR}/.engine_build"
mkdir -p "${BUILD_DIR}"
exec 9>"${BUILD_DIR}/setup.lock"
flock 9

STAGING_DIR="$(mktemp -d "${BUILD_DIR}/setup.XXXXXX")"
trap 'rm -rf "${STAGING_DIR}"' EXIT

for file in graspgenx_everything.onnx graspgenx_everything.onnx.data model_manifest.json; do
  if [[ ! -s ${MODEL_DIR}/${file} ]]; then
    echo "ERROR: ${MODEL_DIR}/${file} is missing or empty. Run export_huggingface_model.py" \
      "first, or set ISAAC_ROS_GRASPGENX_MODEL_DIR to its complete ONNX export." >&2
    exit 1
  fi
done

model_manifest_checksums() {
  if [[ -x ${SCRIPT_DIR}/model_manifest_checksums ]]; then
    # Hermetic py_binary staged alongside this script in the bazel runfiles.
    "${SCRIPT_DIR}/model_manifest_checksums" "$1"
  else
    python3 "${SCRIPT_DIR}/model_manifest_checksums.py" "$1"
  fi
}

# Accept only the expected graph/weights and their SHA-256 checksums.
model_manifest_checksums "${MODEL_DIR}/model_manifest.json" >"${STAGING_DIR}/model.sha256"
(cd "${MODEL_DIR}" && sha256sum --check "${STAGING_DIR}/model.sha256")

if [[ -n ${ISAAC_ROS_GRASPGENX_PLUGIN:-} ]]; then
  PLUGIN="${ISAAC_ROS_GRASPGENX_PLUGIN}"
else
  PACKAGE_DIR="${SCRIPT_DIR}/.."
  if [[ ! -f ${PACKAGE_DIR}/CMakeLists.txt ]]; then
    PACKAGE_DIR="$(ros2 pkg prefix --share isaac_ros_graspgenx_models_install)"
  fi
  cmake -S "${PACKAGE_DIR}" -B "${BUILD_DIR}/native_plugin" \
    -DGRASPGENX_BUILD_PLUGIN_ONLY=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
  cmake --build "${BUILD_DIR}/native_plugin" --parallel
  PLUGIN="${BUILD_DIR}/native_plugin/libgraspgenx_plugins.so"
fi

hash_file() {
  sha256sum <"$1" | cut -d ' ' -f 1
}

TRTEXEC="$(command -v "${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec}")"
ENGINE="${MODELS_DIR}/graspgenx_everything_fp16.engine"
INSTALLED_PLUGIN="${MODELS_DIR}/libgraspgenx_plugins.so"
METADATA="${MODELS_DIR}/engine_build.json"
MODEL_HASH="$(hash_file "${MODEL_DIR}/model_manifest.json")"
PLUGIN_HASH="$(hash_file "${PLUGIN}")"
SCRIPT_HASH="$(hash_file "${BASH_SOURCE[0]}")"
TRTEXEC_HASH="$(hash_file "${TRTEXEC}")"
GPU_INFO="$(nvidia-smi --query-gpu=uuid,name,driver_version,compute_cap --format=csv,noheader)"
BUILD_SIGNATURE="$(printf '%s\n' "${MODEL_HASH}" "${PLUGIN_HASH}" "${SCRIPT_HASH}" \
  "${TRTEXEC_HASH}" "${GPU_INFO}" "${CUDA_VISIBLE_DEVICES:-}" "${CUDA_DEVICE_ORDER:-}" \
  | sha256sum | cut -d ' ' -f 1)"

engine_metadata() {
  printf '{"build_signature": "%s", "engine_sha256": "%s"}\n' "${BUILD_SIGNATURE}" "$1"
}

if [[ ${FORCE_BUILD} == false && -s ${ENGINE} && -s ${INSTALLED_PLUGIN} && -s ${METADATA} &&
  $(<"${METADATA}") == "$(engine_metadata "$(hash_file "${ENGINE}")")" &&
  $(hash_file "${INSTALLED_PLUGIN}") == "${PLUGIN_HASH}" ]]; then
  echo "Reusing ${ENGINE}"
  exit 0
fi

# Random trtexec inputs are invalid plugin indices; build without running inference.
# Bazel drops action output above 1 MiB, so repeat only the TensorRT errors on stderr.
TRTEXEC_LOG="${STAGING_DIR}/trtexec.log"
if ! "${TRTEXEC}" --onnx="${MODEL_DIR}/graspgenx_everything.onnx" \
  --saveEngine="${STAGING_DIR}/graspgenx_everything_fp16.engine" \
  --staticPlugins="${PLUGIN}" --fp16 --skipInference 2>&1 | tee "${TRTEXEC_LOG}"; then
  echo "ERROR: trtexec failed to build ${MODEL_DIR}/graspgenx_everything.onnx on" \
    "${GPU_INFO}. Last TensorRT errors:" >&2
  grep -F --no-group-separator -A 2 '[E]' "${TRTEXEC_LOG}" | tail -n 60 >&2 \
    || tail -n 40 "${TRTEXEC_LOG}" >&2
  exit 1
fi
if [[ ! -s ${STAGING_DIR}/graspgenx_everything_fp16.engine ]]; then
  echo "ERROR: TensorRT did not produce a nonempty engine." >&2
  exit 1
fi
install -m 755 "${PLUGIN}" "${STAGING_DIR}/libgraspgenx_plugins.so"
engine_metadata "$(hash_file "${STAGING_DIR}/graspgenx_everything_fp16.engine")" \
  >"${STAGING_DIR}/engine_build.json"
# Replace runtime assets only after conversion completes successfully.
mv "${STAGING_DIR}/graspgenx_everything_fp16.engine" "${ENGINE}"
mv "${STAGING_DIR}/libgraspgenx_plugins.so" "${INSTALLED_PLUGIN}"
mv "${STAGING_DIR}/engine_build.json" "${METADATA}"
echo "GraspGenX engine and plugin installed in ${MODELS_DIR}."
