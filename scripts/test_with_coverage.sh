#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ctest gcovr clang-format llvm-cov llvm-profdata gcc g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

cmake -S . -B build/coverage -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="--coverage -O0" \
  -DCMAKE_CXX_FLAGS="--coverage -O0" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"

cmake --build build/coverage --parallel
ctest --test-dir build/coverage --output-on-failure -R emel_tests

gcovr \
  --root . \
  --filter src \
  --exclude tests \
  --txt-summary \
  --print-summary \
  --fail-under-line 90 \
  build/coverage
