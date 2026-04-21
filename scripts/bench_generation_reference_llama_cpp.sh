#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"
BUILD_DIR="${EMEL_GENERATION_REFERENCE_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
BUILD_ONLY=false
RUN_ONLY=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_generation_reference_llama_cpp.sh [--build-only] [--run-only] [--zig|--system]

Configures the maintained bench build and runs the built-in llama.cpp reference
generation lane through `bench_runner --mode=reference`.

Environment:
  EMEL_GENERATION_REFERENCE_BUILD_DIR  override build directory
  BENCH_REF_OVERRIDE                   override fetched llama.cpp ref
  EMEL_BENCH_SUITE                     defaults to generation for this wrapper
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=true ;;
    --run-only) RUN_ONLY=true ;;
    --zig) USE_ZIG=true ;;
    --system) USE_ZIG=false ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage
      exit 1
      ;;
  esac
done

if $BUILD_ONLY && $RUN_ONLY; then
  echo "error: --build-only and --run-only are mutually exclusive" >&2
  exit 1
fi

if ! $RUN_ONLY; then
  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
fi

bench_cc="${BENCH_CC:-cc}"
bench_cxx="${BENCH_CXX:-c++}"
bench_cc_arg=""
bench_cxx_arg=""
bench_asm_arg=""
bench_c_flags=""
bench_cxx_flags=""
if $USE_ZIG && ! $RUN_ONLY; then
  if ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi
  bench_cc="$(command -v zig)"
  bench_cxx="$bench_cc"
  bench_cc_arg="cc"
  bench_cxx_arg="c++"
  bench_asm_arg="cc"
  bench_c_flags="-fno-sanitize=undefined"
  bench_cxx_flags="-fno-sanitize=undefined"
fi

if ! $RUN_ONLY; then
  ref_file="$TOOLS_DIR/reference_ref.txt"
  ref_value=""
  if [[ -f "$ref_file" ]]; then
    ref_value="$(head -n 1 "$ref_file" | tr -d '[:space:]')"
  fi
  if [[ -n "${BENCH_REF_OVERRIDE:-}" ]]; then
    ref_value="${BENCH_REF_OVERRIDE}"
  fi
  if [[ -z "$ref_value" ]]; then
    ref_value="master"
  fi

  cmake_args=(
    -S "$TOOLS_DIR"
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DEMEL_ENABLE_TESTS=OFF
    -DREF_IMPL_REF="$ref_value"
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
    "-DCMAKE_ASM_COMPILER=$bench_cc"
  )
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_asm_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ -n "$bench_c_flags" ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$bench_c_flags")
  fi
  if [[ -n "$bench_cxx_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$bench_cxx_flags")
  fi

  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" --parallel --target bench_runner
fi

if $BUILD_ONLY; then
  exit 0
fi

export EMEL_BENCH_SUITE="${EMEL_BENCH_SUITE:-generation}"
"$BUILD_DIR/bench_runner" --mode=reference
