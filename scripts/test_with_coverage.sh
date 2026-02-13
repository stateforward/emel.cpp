#!/usr/bin/env bash
set -euo pipefail

# Resolve Homebrew LLVM when binaries exist but are not in PATH.
if ! command -v llvm-cov >/dev/null 2>&1 || ! command -v llvm-profdata >/dev/null 2>&1; then
  for llvm_bin in /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin; do
    if [ -x "$llvm_bin/llvm-cov" ] && [ -x "$llvm_bin/llvm-profdata" ]; then
      export PATH="$llvm_bin:$PATH"
      break
    fi
  done
fi

for tool in cmake ctest gcovr clang-format llvm-cov llvm-profdata gcc g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

rm -rf build/coverage

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
