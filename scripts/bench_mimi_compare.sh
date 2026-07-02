#!/usr/bin/env bash
# Two-lane Mimi codec comparison (surface mimi_compare/v1).
#
# Builds the EMEL lane (mimi_emel_parity_runner + bench suite) and the
# reference lane (moshi_reference_driver: pinned moshi.cpp + ggml +
# SentencePiece) via the speech_codec_mimi bench-tools configure, fetches
# the pinned real Mimi artifacts, and runs tools/bench/mimi_compare.py.
#
# Honest label: the reference executes ggml's f16 conv pipeline while the
# EMEL lane computes in f32, so this reports code-match similarity + decode
# PSNR without claiming kernel parity (see mimi_compare.py --help).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_MIMI_COMPARE_BUILD_DIR:-$ROOT_DIR/build/mimi_compare_tools}"
OUTPUT_DIR="${EMEL_MIMI_COMPARE_OUTPUT_DIR:-$ROOT_DIR/build/mimi_compare}"
ARTIFACT_DIR="${EMEL_MOSHI_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/moshi_reference}"
BUILD_ONLY=false
RUN_ONLY=false

usage() {
  cat <<'USAGE'
usage: scripts/bench_mimi_compare.sh [--build-only|--run-only] [--output-dir DIR]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

for tool in cmake ninja python3 git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if ! $RUN_ONLY; then
  # full mode: fetch + sha256 verify + run the emel converter (no build)
  "$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh"

  cmake -S "$ROOT_DIR/tools/bench" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMEL_BENCH_SUITE_FILTER=speech_codec_mimi
  cmake --build "$BUILD_DIR" --parallel \
    --target mimi_emel_parity_runner moshi_reference_driver bench_runner
fi

if $BUILD_ONLY; then
  echo "mimi compare lanes built in $BUILD_DIR"
  exit 0
fi

EMEL_RUNNER="$BUILD_DIR/mimi_emel_parity_runner"
REFERENCE_DRIVER="$BUILD_DIR/moshi_reference_driver"
EMEL_MODEL="$ARTIFACT_DIR/mimi-e351c8d8-125-emel.gguf"
REFERENCE_MODEL="$ARTIFACT_DIR/mimi-e351c8d8-125.gguf"
for path in "$EMEL_RUNNER" "$REFERENCE_DRIVER" "$EMEL_MODEL" "$REFERENCE_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "error: missing artifact: $path (run without --run-only first)" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
AUDIO="$OUTPUT_DIR/tone_440_24k.wav"
python3 - "$AUDIO" <<'PY'
import math, struct, sys, wave
with wave.open(sys.argv[1], "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    for index in range(24000):
        value = 0.15 * math.sin(2.0 * math.pi * 440.0 * (index / 24000))
        wav.writeframesraw(struct.pack("<h", int(value * 32767)))
PY

python3 "$ROOT_DIR/tools/bench/mimi_compare.py" \
  --emel-runner "$EMEL_RUNNER" \
  --reference-driver "$REFERENCE_DRIVER" \
  --emel-model "$EMEL_MODEL" \
  --reference-model "$REFERENCE_MODEL" \
  --audio "$AUDIO" \
  --compare-decode \
  --json-out "$OUTPUT_DIR/mimi_compare.json"

echo "report: $OUTPUT_DIR/mimi_compare.json"
