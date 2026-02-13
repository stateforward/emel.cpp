#!/usr/bin/env bash
set -euo pipefail

if ! command -v zig >/dev/null 2>&1; then
  echo "error: zig not found" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found" >&2
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "error: ninja not found" >&2
  exit 1
fi

zig_bin="$(command -v zig)"

cmake -S . -B build/zig -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$zig_bin" \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER="$zig_bin" \
  -DCMAKE_CXX_COMPILER_ARG1=c++
cmake --build build/zig --parallel
