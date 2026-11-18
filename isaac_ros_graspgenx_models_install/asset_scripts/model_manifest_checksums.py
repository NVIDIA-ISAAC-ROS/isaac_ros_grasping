#!/usr/bin/env python3
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

"""Print the GraspGenX model checksums from model_manifest.json in sha256sum format."""

import argparse
import json
import re
import sys

EXPECTED_FILES = ('graspgenx_everything.onnx', 'graspgenx_everything.onnx.data')
SHA256_PATTERN = re.compile(r'[0-9a-fA-F]{64}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='Path to model_manifest.json')
    manifest_path = parser.parse_args().manifest

    with open(manifest_path, encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or sorted(files) != sorted(EXPECTED_FILES):
        print(f'ERROR: {manifest_path} must list checksums for exactly '
              f'{", ".join(EXPECTED_FILES)}; found {files!r}', file=sys.stderr)
        return 1
    for name in EXPECTED_FILES:
        checksum = files[name]
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
            print(f'ERROR: {manifest_path} has an invalid SHA-256 for {name}: {checksum!r}',
                  file=sys.stderr)
            return 1
        print(f'{checksum}  {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
