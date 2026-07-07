#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"
BUILD_DIR="${EMEL_GENERATION_REFERENCE_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
BUILD_ONLY=false
RUN_ONLY=false
REFERENCE_THREADS=""
REFERENCE_DECODE_THREADS="${EMEL_BENCH_GENERATION_REFERENCE_DECODE_THREADS:-}"
REFERENCE_BATCH_THREADS="${EMEL_BENCH_GENERATION_REFERENCE_BATCH_THREADS:-}"
BENCHMARK_LANE_SELECTOR="${EMEL_BENCH_GENERATION_LANE:-${EMEL_BENCH_GENERATION_LANES:-both}}"
USE_ZIG=true
case "${EMEL_GENERATION_REFERENCE_COMPILER_MODE:-}" in
  "")
    ;;
  zig)
    USE_ZIG=true
    ;;
  system)
    USE_ZIG=false
    ;;
  *)
    echo "error: EMEL_GENERATION_REFERENCE_COMPILER_MODE must be 'zig' or 'system'" >&2
    exit 1
    ;;
esac

usage() {
  cat <<'USAGE'
usage: scripts/bench_generation_reference_llama_cpp.sh [--build-only] [--run-only] [--reference-threads N] [--reference-decode-threads N] [--reference-batch-threads N] [--benchmark-lane single|multithreaded|both] [--zig|--system]

Configures the maintained bench build and runs the built-in llama.cpp reference
generation lane through `bench_runner --mode=reference`.

Environment:
  EMEL_GENERATION_REFERENCE_BUILD_DIR  override build directory
  EMEL_GENERATION_REFERENCE_COMPILER_MODE zig or system default compiler mode
  EMEL_BENCH_GENERATION_LANES          single, multithreaded, or both (default both)
  EMEL_BENCH_GENERATION_REFERENCE_THREADS legacy default for llama.cpp decode and batch threads
  EMEL_BENCH_GENERATION_REFERENCE_DECODE_THREADS llama.cpp n_threads override
  EMEL_BENCH_GENERATION_REFERENCE_BATCH_THREADS llama.cpp n_threads_batch override
  BENCH_REF_OVERRIDE                   override fetched llama.cpp ref
  EMEL_BENCH_SUITE                     defaults to generation for this wrapper
USAGE
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $name must be a positive integer" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)
      BUILD_ONLY=true
      shift
      ;;
    --run-only)
      RUN_ONLY=true
      shift
      ;;
    --reference-threads)
      REFERENCE_THREADS="${2:-}"
      shift 2
      ;;
    --reference-decode-threads)
      REFERENCE_DECODE_THREADS="${2:-}"
      shift 2
      ;;
    --reference-batch-threads)
      REFERENCE_BATCH_THREADS="${2:-}"
      shift 2
      ;;
    --benchmark-lane)
      BENCHMARK_LANE_SELECTOR="${2:-}"
      shift 2
      ;;
    --single-only)
      BENCHMARK_LANE_SELECTOR="single"
      shift
      ;;
    --multithreaded-only)
      BENCHMARK_LANE_SELECTOR="multithreaded"
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

if $BUILD_ONLY && $RUN_ONLY; then
  echo "error: --build-only and --run-only are mutually exclusive" >&2
  exit 1
fi

validate_positive_integer "--reference-threads" "$REFERENCE_THREADS"
validate_positive_integer "--reference-decode-threads" "$REFERENCE_DECODE_THREADS"
validate_positive_integer "--reference-batch-threads" "$REFERENCE_BATCH_THREADS"

selected_lanes=()
case "$BENCHMARK_LANE_SELECTOR" in
  both|single,multithreaded|multithreaded,single)
    selected_lanes=(single multithreaded)
    ;;
  single)
    selected_lanes=(single)
    ;;
  multithreaded)
    selected_lanes=(multithreaded)
    ;;
  *)
    echo "error: --benchmark-lane/EMEL_BENCH_GENERATION_LANES must be single, multithreaded, or both" >&2
    exit 1
    ;;
esac

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
for benchmark_lane in "${selected_lanes[@]}"; do
  lane_reference_threads="$REFERENCE_THREADS"
  if [[ "$benchmark_lane" == "single" ]]; then
    lane_reference_threads="1"
  elif [[ -z "$lane_reference_threads" ]]; then
    lane_reference_threads="${EMEL_BENCH_GENERATION_REFERENCE_THREADS:-${EMEL_BENCH_REFERENCE_THREADS:-8}}"
  fi
  lane_reference_decode_threads="${REFERENCE_DECODE_THREADS:-$lane_reference_threads}"
  lane_reference_batch_threads="${REFERENCE_BATCH_THREADS:-$lane_reference_threads}"

  EMEL_BENCH_GENERATION_LANE="$benchmark_lane" \
    EMEL_BENCH_GENERATION_REFERENCE_THREADS="$lane_reference_threads" \
    EMEL_BENCH_GENERATION_REFERENCE_DECODE_THREADS="$lane_reference_decode_threads" \
    EMEL_BENCH_GENERATION_REFERENCE_BATCH_THREADS="$lane_reference_batch_threads" \
    "$BUILD_DIR/bench_runner" --mode=reference
done
