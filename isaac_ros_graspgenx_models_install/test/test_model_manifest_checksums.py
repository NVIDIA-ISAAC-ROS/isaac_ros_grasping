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

"""Tests for the GraspGenX model manifest checksum reader."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parent.parent / 'asset_scripts' / 'model_manifest_checksums.py'
ONNX_SHA256 = 'a' * 64
DATA_SHA256 = 'B' * 64


def _run(tmp_path: Path, manifest) -> subprocess.CompletedProcess:
    manifest_path = tmp_path / 'model_manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest_path)],
        capture_output=True, text=True, check=False)


def test_prints_checksums_in_sha256sum_format(tmp_path):
    result = _run(tmp_path, {'files': {
        'graspgenx_everything.onnx.data': DATA_SHA256,
        'graspgenx_everything.onnx': ONNX_SHA256,
    }})
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        f'{ONNX_SHA256}  graspgenx_everything.onnx\n'
        f'{DATA_SHA256}  graspgenx_everything.onnx.data\n')


@pytest.mark.parametrize('manifest', [
    [],
    {},
    {'files': []},
    {'files': {'graspgenx_everything.onnx': ONNX_SHA256}},
    {'files': {
        'graspgenx_everything.onnx': ONNX_SHA256,
        'graspgenx_everything.onnx.data': DATA_SHA256,
        'extra.onnx': ONNX_SHA256,
    }},
], ids=['not_object', 'no_files', 'files_not_object', 'missing_weights', 'extra_file'])
def test_rejects_unexpected_file_list(tmp_path, manifest):
    result = _run(tmp_path, manifest)
    assert result.returncode == 1
    assert result.stdout == ''
    assert 'must list checksums for exactly' in result.stderr


@pytest.mark.parametrize('checksum', ['a' * 63, 'g' * 64, 64], ids=['short', 'not_hex', 'number'])
def test_rejects_invalid_checksum(tmp_path, checksum):
    result = _run(tmp_path, {'files': {
        'graspgenx_everything.onnx': ONNX_SHA256,
        'graspgenx_everything.onnx.data': checksum,
    }})
    assert result.returncode == 1
    assert 'invalid SHA-256 for graspgenx_everything.onnx.data' in result.stderr


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
