#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ctest ninja zig; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/paritychecker_zig"
zig_bin="$(command -v zig)"

cmake -S "$ROOT_DIR/tools/paritychecker" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$zig_bin" \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER="$zig_bin" \
  -DCMAKE_CXX_COMPILER_ARG1=c++ \
  -DGGML_METAL=OFF \
  -DLLAMA_METAL=OFF

cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure -R paritychecker_tests
