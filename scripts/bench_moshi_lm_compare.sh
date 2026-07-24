#!/usr/bin/env bash
# PersonaPlex Moshi LM contract benchmark.
#
# This is an EMEL-only load/contract lane: it catches regressions where the
# maintained Moshi model loader rejects the converted PersonaPlex Q4_K LM GGUF.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/build_jobs.sh
source "$ROOT_DIR/scripts/build_jobs.sh"

BUILD_DIR_OVERRIDE="${EMEL_MOSHI_LM_COMPARE_BUILD_DIR:-}"
BUILD_DIR=""
MODEL_PATH=""
MODEL_EXPLICIT=false
BUILD_ONLY=false
RUN_ONLY=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_moshi_lm_compare.sh [--build-only|--run-only] [--model PATH]
                                        [--zig|--system]

Runs the speech_lm_moshi EMEL load-contract benchmark for the converted
PersonaPlex Q4_K Moshi LM GGUF. Set EMEL_PERSONAPLEX_LM_MODEL,
EMEL_MOSHI_LM_MODEL, EMEL_BENCH_SPEECH_LM_MOSHI_MODEL,
EMEL_MOSHI_REFERENCE_MODEL_EMEL, or pass --model to override the model path.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    --model)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "error: --model requires a path" >&2
        exit 1
      fi
      MODEL_PATH="$2"
      MODEL_EXPLICIT=true
      shift 2
      ;;
    --zig) USE_ZIG=true; shift ;;
    --system) USE_ZIG=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

if $BUILD_ONLY && $RUN_ONLY; then
  echo "error: --build-only and --run-only are mutually exclusive" >&2
  exit 1
fi

if [[ -n "$BUILD_DIR_OVERRIDE" ]]; then
  BUILD_DIR="$BUILD_DIR_OVERRIDE"
elif $USE_ZIG; then
  BUILD_DIR="$ROOT_DIR/build/moshi_lm_compare_tools_zig"
else
  BUILD_DIR="$ROOT_DIR/build/moshi_lm_compare_tools_system"
fi

if ! $BUILD_ONLY; then
  if ! $MODEL_EXPLICIT; then
    for env_name in \
      EMEL_PERSONAPLEX_LM_MODEL \
      EMEL_MOSHI_LM_MODEL \
      EMEL_BENCH_SPEECH_LM_MOSHI_MODEL \
      EMEL_MOSHI_REFERENCE_MODEL_EMEL; do
      env_value="${!env_name:-}"
      if [[ -n "$env_value" ]]; then
        MODEL_PATH="$env_value"
        break
      fi
    done
  fi

  if [[ -z "$MODEL_PATH" && ! $RUN_ONLY ]]; then
    setup_output="$("$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh")"
    MODEL_PATH="$(
      printf '%s\n' "$setup_output" |
        awk -F= '$1 == "EMEL_MOSHI_REFERENCE_MODEL_EMEL" {
          sub(/^[^=]*=/, "");
          print;
          exit;
        }'
    )"
  fi

  if [[ -z "$MODEL_PATH" ]]; then
    for candidate in \
      "$ROOT_DIR/build/moshi_reference/model-q4_k-emel.gguf" \
      "$ROOT_DIR/../companion/zig-out/personaplex-emel-converted/Codes4Fun/personaplex-7b-v1-q4_k-GGUF/model-q4_k.gguf" \
      "$ROOT_DIR/../../companion/zig-out/personaplex-emel-converted/Codes4Fun/personaplex-7b-v1-q4_k-GGUF/model-q4_k.gguf" \
      "$ROOT_DIR/../../../companion/zig-out/personaplex-emel-converted/Codes4Fun/personaplex-7b-v1-q4_k-GGUF/model-q4_k.gguf"; do
      if [[ -f "$candidate" ]]; then
        MODEL_PATH="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    echo "error: missing PersonaPlex Moshi LM GGUF; pass --model or set" \
         "EMEL_PERSONAPLEX_LM_MODEL, EMEL_MOSHI_LM_MODEL, or" \
         "EMEL_BENCH_SPEECH_LM_MOSHI_MODEL, or" \
         "EMEL_MOSHI_REFERENCE_MODEL_EMEL" >&2
    exit 1
  fi
fi

for tool in cmake ninja git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if ! $RUN_ONLY && $USE_ZIG; then
  # shellcheck source=scripts/zig_toolchain.sh
  source "$ROOT_DIR/scripts/zig_toolchain.sh"
fi

cmake_args=(-S "$ROOT_DIR/tools/bench" -B "$BUILD_DIR" -G Ninja
            -DCMAKE_BUILD_TYPE=Release
            -DEMEL_BENCH_SUITE_FILTER=speech_lm_moshi
            -DEMEL_BENCH_SKIP_MOSHI_REFERENCE=ON)

if ! $RUN_ONLY && $USE_ZIG; then
  if ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi
  zig_bin="$(command -v zig)"
  cmake_args+=("-DCMAKE_C_COMPILER=$zig_bin"
               "-DCMAKE_C_COMPILER_ARG1=cc"
               "-DCMAKE_CXX_COMPILER=$zig_bin"
               "-DCMAKE_CXX_COMPILER_ARG1=c++"
               "-DCMAKE_ASM_COMPILER=$zig_bin"
               "-DCMAKE_ASM_COMPILER_ARG1=cc"
               "-DCMAKE_C_FLAGS=-fno-sanitize=undefined"
               "-DCMAKE_CXX_FLAGS=-fno-sanitize=undefined")
  cmake_args+=("${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}")
fi

if ! $RUN_ONLY; then
  cmake "${cmake_args[@]}" >&2
  cmake --build "$BUILD_DIR" --parallel "$EMEL_BUILD_JOBS" --target bench_runner >&2
fi

if $BUILD_ONLY; then
  echo "moshi lm bench built in $BUILD_DIR"
  exit 0
fi

EMEL_BENCH_SPEECH_LM_MOSHI=1 \
EMEL_PERSONAPLEX_LM_MODEL="$MODEL_PATH" \
EMEL_MOSHI_LM_MODEL="$MODEL_PATH" \
EMEL_BENCH_SPEECH_LM_MOSHI_MODEL="$MODEL_PATH" \
EMEL_BENCH_SUITE=speech_lm_moshi \
EMEL_BENCH_ITERS="${EMEL_MOSHI_LM_BENCH_ITERS:-1}" \
EMEL_BENCH_RUNS="${EMEL_MOSHI_LM_BENCH_RUNS:-1}" \
EMEL_BENCH_WARMUP_ITERS="${EMEL_MOSHI_LM_BENCH_WARMUP_ITERS:-0}" \
EMEL_BENCH_WARMUP_RUNS="${EMEL_MOSHI_LM_BENCH_WARMUP_RUNS:-0}" \
  "$BUILD_DIR/bench_runner" --mode=emel
