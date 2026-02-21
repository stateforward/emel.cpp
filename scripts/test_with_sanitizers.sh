#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ctest ninja clang clang++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="$ROOT_DIR/build/sanitizers"

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
ctest_jobs=$((cpu_count / 2))
if [[ "$ctest_jobs" -lt 1 ]]; then
  ctest_jobs=1
fi

run_sanitizer() {
  local name="$1"
  local c_flags="$2"
  local cxx_flags="$3"
  local ld_flags="$4"
  shift 4
  local build_dir="$BUILD_ROOT/$name"
  rm -rf "$build_dir"
  cmake -S "$ROOT_DIR" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_EXE_LINKER_FLAGS="$ld_flags"
  cmake --build "$build_dir" --parallel
  env "$@" ctest --test-dir "$build_dir" --output-on-failure -R emel_tests -j "$ctest_jobs"
}

run_sanitizer "asan_ubsan" \
  "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined" \
  "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined" \
  "-fsanitize=address,undefined" \
  ASAN_OPTIONS=halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1

run_sanitizer "tsan" \
  "-O1 -g -fno-omit-frame-pointer -fsanitize=thread" \
  "-O1 -g -fno-omit-frame-pointer -fsanitize=thread" \
  "-fsanitize=thread" \
  TSAN_OPTIONS=halt_on_error=1
