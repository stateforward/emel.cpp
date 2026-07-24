#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/build_jobs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_jobs.sh"
# shellcheck source=scripts/zig_toolchain.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zig_toolchain.sh"

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
test_shards="${EMEL_ZIG_TEST_SHARDS:-}"

cmake -S . -B build/zig -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$zig_bin" \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER="$zig_bin" \
  -DCMAKE_CXX_COMPILER_ARG1=c++ \
  "${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}" \
  -DEMEL_TEST_SHARDS="$test_shards"
cmake --build build/zig --parallel "$EMEL_BUILD_JOBS"

if [[ -n "$EMEL_ZIG_MACOS_DEPLOYMENT_TARGET" ]]; then
  "$PWD/scripts/check_macos_binary_target.sh" \
    "$PWD/build/zig/emel_tests_bin" \
    "$EMEL_ZIG_MACOS_DEPLOYMENT_TARGET"
  if ! "$PWD/build/zig/emel_tests_bin" --list-test-cases >/dev/null; then
    echo "error: Zig test binary failed the macOS launch probe" >&2
    exit 1
  fi
  echo "macOS launch verified: $PWD/build/zig/emel_tests_bin"
fi
