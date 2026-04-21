#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"
BUILD_DIR="${EMEL_BENCH_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
OUTPUT_DIR="${EMEL_GENERATION_COMPARE_OUTPUT_DIR:-$ROOT_DIR/build/generation_compare}"
REFERENCE_BACKEND=""
WORKLOAD_ID=""
SKIP_EMEL_BUILD=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_generation_compare.sh --reference-backend BACKEND [--workload-id ID] [--output-dir DIR] [--skip-emel-build] [--zig|--system]

Builds the maintained EMEL generation benchmark runner, then runs the unified
generation compare workflow against the selected pluggable reference backend manifest.

Environment:
  EMEL_BENCH_BUILD_DIR              override EMEL bench build directory
  EMEL_GENERATION_COMPARE_OUTPUT_DIR override compare output directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference-backend)
      REFERENCE_BACKEND="${2:-}"
      shift 2
      ;;
    --workload-id)
      WORKLOAD_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --skip-emel-build)
      SKIP_EMEL_BUILD=true
      shift
      ;;
    --zig)
      USE_ZIG=true
      shift
      ;;
    --system)
      USE_ZIG=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$REFERENCE_BACKEND" ]]; then
  echo "error: --reference-backend is required" >&2
  usage
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: required tool missing: python3" >&2
  exit 1
fi

if ! $SKIP_EMEL_BUILD; then
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
if $USE_ZIG && ! $SKIP_EMEL_BUILD; then
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

if ! $SKIP_EMEL_BUILD; then
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

compare_args=(
  "$ROOT_DIR/tools/bench/generation_compare.py"
  --reference-backend "$REFERENCE_BACKEND"
  --emel-runner "$BUILD_DIR/bench_runner"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$WORKLOAD_ID" ]]; then
  compare_args+=(--workload-id "$WORKLOAD_ID")
fi

python3 "${compare_args[@]}"
