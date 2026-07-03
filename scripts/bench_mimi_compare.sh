#!/usr/bin/env bash
# Two-lane Mimi codec comparison (surface mimi_compare/v1).
#
# Builds the EMEL lane (mimi_emel_parity_runner + bench suite) and the
# reference lane (moshi_reference_driver: pinned moshi.cpp + ggml +
# SentencePiece) via the speech_codec_mimi bench-tools configure, fetches
# the pinned real Mimi artifacts, and runs tools/bench/mimi_compare.py.
#
# The EMEL lane consumes the same effective operand pipeline as the
# reference (f16 convs, bf16 KV attention, exact ggml numerics), so encode
# comparison is gated TOKEN-EXACT.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_MIMI_COMPARE_BUILD_DIR:-$ROOT_DIR/build/mimi_compare_tools}"
OUTPUT_DIR="${EMEL_MIMI_COMPARE_OUTPUT_DIR:-$ROOT_DIR/build/mimi_compare}"
ARTIFACT_DIR="${EMEL_MOSHI_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/moshi_reference}"
MLX_ARTIFACT_DIR="${EMEL_PERSONAPLEX_MLX_ARTIFACT_DIR:-$ROOT_DIR/build/personaplex_mlx_reference}"
BUILD_ONLY=false
RUN_ONLY=false
REFERENCE_BACKEND="moshi_cpp"

usage() {
  cat <<'USAGE'
usage: scripts/bench_mimi_compare.sh [--build-only|--run-only] [--output-dir DIR]
                                     [--reference=moshi_cpp|personaplex-mlx]

  --reference=moshi_cpp       pinned moshi.cpp driver; encode gated
                              token-exact (default)
  --reference=personaplex-mlx pinned personaplex-mlx MLX Mimi driver
                              (Apple Silicon); cross-implementation float
                              lane, gated on code-match fraction
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --reference=moshi_cpp|--reference=moshi-cpp) REFERENCE_BACKEND="moshi_cpp"; shift ;;
    --reference=personaplex_mlx|--reference=personaplex-mlx) REFERENCE_BACKEND="personaplex_mlx"; shift ;;
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
  if [[ "$REFERENCE_BACKEND" == "personaplex_mlx" ]]; then
    "$ROOT_DIR/scripts/setup_personaplex_mlx_reference.sh"
    reference_targets=()
  else
    "$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh"
    reference_targets=(moshi_reference_driver)
  fi

  cmake -S "$ROOT_DIR/tools/bench" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DEMEL_BENCH_SUITE_FILTER=speech_codec_mimi
  cmake --build "$BUILD_DIR" --parallel \
    --target mimi_emel_parity_runner bench_runner \
    ${reference_targets[@]+"${reference_targets[@]}"}
fi

if $BUILD_ONLY; then
  echo "mimi compare lanes built in $BUILD_DIR"
  exit 0
fi

EMEL_RUNNER="$BUILD_DIR/mimi_emel_parity_runner"
EMEL_MODEL="$ARTIFACT_DIR/mimi-e351c8d8-125-emel.gguf"
if [[ "$REFERENCE_BACKEND" == "personaplex_mlx" ]]; then
  REFERENCE_DRIVER="$MLX_ARTIFACT_DIR/mimi_driver"
  REFERENCE_MODEL="$MLX_ARTIFACT_DIR/tokenizer-e351c8d8-checkpoint125.safetensors"
else
  REFERENCE_DRIVER="$BUILD_DIR/moshi_reference_driver"
  REFERENCE_MODEL="$ARTIFACT_DIR/mimi-e351c8d8-125.gguf"
fi
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

if [[ "$REFERENCE_BACKEND" == "personaplex_mlx" ]]; then
  # Cross-implementation float lane (MLX Metal numerics): deep RVQ layers
  # quantize near-zero residuals, so their codes cascade apart between any
  # two float implementations (measured on the 440 Hz tone: EMEL vs MLX
  # 0.52 full-code match while Kyutai's own rustymimi vs MLX is 0.66).
  # Gate the numerically stable early streams hard and keep a sanity floor
  # on the full match; token exactness remains the moshi_cpp lane's gate.
  # The timing pass is measurement-only: each lane self-reports steady-state
  # per-frame loop time over a 10 s signal, excluding model load.
  TIMING_AUDIO="$OUTPUT_DIR/tone_440_24k_10s.wav"
  python3 - "$TIMING_AUDIO" <<'PY'
import math, struct, sys, wave
with wave.open(sys.argv[1], "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    for index in range(240000):
        value = 0.15 * math.sin(2.0 * math.pi * 440.0 * (index / 24000))
        wav.writeframesraw(struct.pack("<h", int(value * 32767)))
PY
  REPORT="$OUTPUT_DIR/mimi_compare_personaplex_mlx.json"
  python3 "$ROOT_DIR/tools/bench/mimi_compare.py" \
    --emel-runner "$EMEL_RUNNER" \
    --reference-driver "$REFERENCE_DRIVER" \
    --emel-model "$EMEL_MODEL" \
    --reference-model "$REFERENCE_MODEL" \
    --audio "$AUDIO" \
    --compare-decode \
    --prefix-streams "${EMEL_MIMI_MLX_PREFIX_STREAMS:-4}" \
    --min-prefix-match "${EMEL_MIMI_MLX_MIN_PREFIX_MATCH:-0.90}" \
    --min-code-match "${EMEL_MIMI_MLX_MIN_CODE_MATCH:-0.40}" \
    --timing-audio "$TIMING_AUDIO" \
    --reference-label "personaplex-mlx" \
    --json-out "$REPORT"
else
  REPORT="$OUTPUT_DIR/mimi_compare.json"
  python3 "$ROOT_DIR/tools/bench/mimi_compare.py" \
    --emel-runner "$EMEL_RUNNER" \
    --reference-driver "$REFERENCE_DRIVER" \
    --emel-model "$EMEL_MODEL" \
    --reference-model "$REFERENCE_MODEL" \
    --audio "$AUDIO" \
    --compare-decode \
    --require-token-exact \
    --json-out "$REPORT"
fi

echo "report: $REPORT"
