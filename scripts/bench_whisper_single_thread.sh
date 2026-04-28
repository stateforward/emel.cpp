#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_WHISPER_COMPARE_BUILD_DIR:-$ROOT_DIR/build/whisper_compare_tools}"
OUTPUT_DIR="${EMEL_WHISPER_BENCH_OUTPUT_DIR:-$ROOT_DIR/build/whisper_benchmark}"
REFERENCE_BUILD_DIR="${EMEL_WHISPER_CPP_BUILD_DIR:-$ROOT_DIR/build/whisper_cpp_reference/build}"
ARTIFACT_DIR="${EMEL_WHISPER_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/whisper_reference}"
WARMUPS="${EMEL_WHISPER_BENCH_WARMUPS:-1}"
ITERATIONS="${EMEL_WHISPER_BENCH_ITERATIONS:-20}"
SKIP_REFERENCE_BUILD=false
SKIP_EMEL_BUILD=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_whisper_single_thread.sh [--output-dir DIR] [--warmups N]
                                              [--iterations N]
                                              [--skip-reference-build]
                                              [--skip-emel-build] [--zig|--system]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --warmups) WARMUPS="${2:-}"; shift 2 ;;
    --iterations) ITERATIONS="${2:-}"; shift 2 ;;
    --skip-reference-build) SKIP_REFERENCE_BUILD=true; shift ;;
    --skip-emel-build) SKIP_EMEL_BUILD=true; shift ;;
    --zig) USE_ZIG=true; shift ;;
    --system) USE_ZIG=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

for tool in cmake ninja python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if ! $SKIP_REFERENCE_BUILD; then
  setup_args=()
  if $USE_ZIG; then setup_args+=(--zig); else setup_args+=(--system); fi
  "$ROOT_DIR/scripts/setup_whisper_cpp_reference.sh" "${setup_args[@]}"
fi

bench_cc="${BENCH_CC:-cc}"
bench_cxx="${BENCH_CXX:-c++}"
bench_cc_arg=""
bench_cxx_arg=""
if $USE_ZIG && ! $SKIP_EMEL_BUILD; then
  if ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi
  bench_cc="$(command -v zig)"
  bench_cxx="$bench_cc"
  bench_cc_arg="cc"
  bench_cxx_arg="c++"
fi

if ! $SKIP_EMEL_BUILD; then
  cmake_args=(
    -S "$ROOT_DIR/tools/bench"
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DEMEL_ENABLE_TESTS=OFF
    -DEMEL_BENCH_SUITE_FILTER=whisper_compare
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
  )
  if [[ -n "$bench_cc_arg" ]]; then cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg"); fi
  if [[ -n "$bench_cxx_arg" ]]; then cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg"); fi
  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" --parallel --target whisper_emel_parity_runner
fi

cli_path="${EMEL_WHISPER_CPP_CLI:-}"
if [[ -z "$cli_path" ]]; then
  cli_path="$(find "$REFERENCE_BUILD_DIR" -type f -perm -111 -name 'whisper-cli' | head -1)"
fi
model_path="${EMEL_WHISPER_REFERENCE_MODEL:-$ARTIFACT_DIR/whisper-tiny-q8_0-whispercpp.gguf}"
audio_path="${EMEL_WHISPER_REFERENCE_AUDIO:-$ARTIFACT_DIR/phase99_440hz_16khz_mono.wav}"
emel_model="${EMEL_WHISPER_EMEL_MODEL:-$model_path}"
tokenizer_model="${EMEL_WHISPER_TOKENIZER_MODEL:-$ROOT_DIR/tests/models/tokenizer-tiny.json}"
runner="${EMEL_WHISPER_EMEL_RUNNER:-$BUILD_DIR/whisper_emel_parity_runner}"

python3 "$ROOT_DIR/tools/bench/whisper_benchmark.py" \
  --output-dir "$OUTPUT_DIR" \
  --emel-runner "$runner" \
  --emel-model "$emel_model" \
  --tokenizer "$tokenizer_model" \
  --reference-cli "$cli_path" \
  --reference-model "$model_path" \
  --audio "$audio_path" \
  --warmups "$WARMUPS" \
  --iterations "$ITERATIONS"
