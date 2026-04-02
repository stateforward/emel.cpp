#!/usr/bin/env bash
set -euo pipefail

LINE_COVERAGE_MIN="${LINE_COVERAGE_MIN:-90}"
BRANCH_COVERAGE_MIN="${BRANCH_COVERAGE_MIN:-50}"
COVERAGE_BUILD_DIR="${EMEL_COVERAGE_BUILD_DIR:-build/coverage}"
COVERAGE_CLEAN="${EMEL_COVERAGE_CLEAN:-0}"

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

if [[ "$COVERAGE_CLEAN" == "1" ]]; then
  rm -rf "$COVERAGE_BUILD_DIR"
fi

cmake -S . -B "$COVERAGE_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="--coverage -O0" \
  -DCMAKE_CXX_FLAGS="--coverage -O0" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"

cmake --build "$COVERAGE_BUILD_DIR" --parallel
cpu_count=2
if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  cpu_count="$(getconf _NPROCESSORS_ONLN || echo 2)"
elif command -v sysctl >/dev/null 2>&1; then
  cpu_count="$(sysctl -n hw.ncpu || echo 2)"
fi
if [[ -z "$cpu_count" || "$cpu_count" -lt 1 ]]; then
  cpu_count=2
fi
ctest_jobs="$cpu_count"

find "$COVERAGE_BUILD_DIR" -name '*.gcda' -delete
find "$COVERAGE_BUILD_DIR" -maxdepth 1 -type d -name 'profiles*' -exec rm -rf {} +

ctest --test-dir "$COVERAGE_BUILD_DIR" --output-on-failure -R '^emel_tests' -j "$ctest_jobs"

echo "enforcing coverage thresholds: line >= ${LINE_COVERAGE_MIN}%, branch >= ${BRANCH_COVERAGE_MIN}%"

gcovr \
  --root . \
  --filter src \
  --exclude tests \
  --exclude 'src/emel/.*/sm.hpp' \
  --gcov-ignore-parse-errors suspicious_hits.warn_once_per_file \
  --exclude-throw-branches \
  --exclude-unreachable-branches \
  --txt-summary \
  --print-summary \
  --fail-under-line "$LINE_COVERAGE_MIN" \
  --fail-under-branch "$BRANCH_COVERAGE_MIN" \
  "$COVERAGE_BUILD_DIR"
