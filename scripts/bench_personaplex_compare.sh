#!/usr/bin/env bash
# CPU-only, fixed-seed PersonaPlex speech comparison using isolated subprocess lanes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/build_jobs.sh
source "$ROOT_DIR/scripts/build_jobs.sh"

BUILD_DIR="${EMEL_PERSONAPLEX_COMPARE_BUILD_DIR:-$ROOT_DIR/build/personaplex_compare_tools_zig}"
OUTPUT_DIR="${EMEL_PERSONAPLEX_COMPARE_OUTPUT_DIR:-$ROOT_DIR/build/personaplex_compare}"
ARTIFACT_DIR="${EMEL_MOSHI_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/moshi_reference}"
AUDIO=""
FRAMES=125
SEED=1234
THREADS=1
BUILD_ONLY=false
RUN_ONLY=false

usage() {
  cat <<'USAGE'
usage: scripts/bench_personaplex_compare.sh [options]

  --audio PATH       24 kHz mono s16 WAV; default uses macOS say for
                     "Hey, I'm Gabe. How are you doing?"
  --frames N         generated frames per lane (default: 125 / 10 seconds)
  --seed N           fixed sampling seed (default: 1234)
  --threads N        CPU threads in moshi.cpp reference lane (default: 1)
  --output-dir PATH  output WAV/log/report directory
  --build-only       fetch/configure/build without running
  --run-only         use an existing build and artifacts
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio) AUDIO="${2:-}"; shift 2 ;;
    --frames) FRAMES="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

if $BUILD_ONLY && $RUN_ONLY; then
  echo "error: --build-only and --run-only are mutually exclusive" >&2
  exit 1
fi
for value in "$FRAMES" "$SEED" "$THREADS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: frames, seed, and threads must be positive integers" >&2
    exit 1
  fi
done
for tool in cmake ninja git python3 zig; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if ! $RUN_ONLY; then
  "$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh" >/dev/null
  zig_bin="$(command -v zig)"
  cmake -S "$ROOT_DIR/tools/bench" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMEL_BENCH_SUITE_FILTER=speech_dialogue_moshi \
    -DEMEL_BENCH_SKIP_MOSHI_REFERENCE=OFF \
    -DCMAKE_C_COMPILER="$zig_bin" \
    -DCMAKE_C_COMPILER_ARG1=cc \
    -DCMAKE_CXX_COMPILER="$zig_bin" \
    -DCMAKE_CXX_COMPILER_ARG1=c++ \
    -DCMAKE_ASM_COMPILER="$zig_bin" \
    -DCMAKE_ASM_COMPILER_ARG1=cc \
    -DCMAKE_C_FLAGS=-fno-sanitize=undefined \
    -DCMAKE_CXX_FLAGS=-fno-sanitize=undefined
  cmake --build "$BUILD_DIR" --parallel "$EMEL_BUILD_JOBS" \
    --target personaplex_emel_runner moshi_reference_driver
fi

if $BUILD_ONLY; then
  echo "PersonaPlex CPU comparison lanes built in $BUILD_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
if [[ -z "$AUDIO" ]]; then
  if ! command -v say >/dev/null 2>&1; then
    echo "error: default organic input needs macOS say; pass --audio" >&2
    exit 1
  fi
  AUDIO="$OUTPUT_DIR/say_gabe_how_are_you_24k.wav"
  say -v Samantha -r 175 --file-format=WAVE --data-format=LEI16@24000 \
    --channels=1 -o "$AUDIO" "Hey, I'm Gabe. How are you doing?"
fi

EMEL_RUNNER="$BUILD_DIR/personaplex_emel_runner"
REFERENCE_DRIVER="$BUILD_DIR/moshi_reference_driver"
EMEL_MIMI="$ARTIFACT_DIR/mimi-e351c8d8-125-personaplex-emel.gguf"
REFERENCE_MIMI="$ARTIFACT_DIR/mimi-e351c8d8-125.gguf"
EMEL_LM="$ARTIFACT_DIR/model-q4_k-emel.gguf"
REFERENCE_LM="$ARTIFACT_DIR/model-q4_k.gguf"
EMEL_VOICE="$ARTIFACT_DIR/voices/NATF0-emel.gguf"
REFERENCE_VOICE="$ARTIFACT_DIR/voices/NATF0.gguf"
CONFIG="$ARTIFACT_DIR/personaplex-config.json"
INFERENCE_CONFIG="$ROOT_DIR/tools/bench/personaplex-inference.json"
for path in "$AUDIO" "$EMEL_RUNNER" "$REFERENCE_DRIVER" "$EMEL_MIMI" \
            "$REFERENCE_MIMI" "$EMEL_LM" "$REFERENCE_LM" "$EMEL_VOICE" \
            "$REFERENCE_VOICE" "$CONFIG" "$INFERENCE_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing required artifact: $path" >&2
    exit 1
  fi
done

python3 "$ROOT_DIR/tools/bench/personaplex_compare.py" \
  --emel-runner "$EMEL_RUNNER" \
  --reference-driver "$REFERENCE_DRIVER" \
  --emel-mimi "$EMEL_MIMI" \
  --reference-mimi "$REFERENCE_MIMI" \
  --emel-lm "$EMEL_LM" \
  --reference-lm "$REFERENCE_LM" \
  --emel-voice "$EMEL_VOICE" \
  --reference-voice "$REFERENCE_VOICE" \
  --config "$CONFIG" \
  --inference-config "$INFERENCE_CONFIG" \
  --audio "$AUDIO" \
  --output-dir "$OUTPUT_DIR" \
  --frames "$FRAMES" \
  --seed "$SEED" \
  --threads "$THREADS"
