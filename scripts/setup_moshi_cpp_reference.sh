#!/usr/bin/env bash
set -euo pipefail

# Fetches the pinned moshi.cpp reference checkout and the pinned PersonaPlex /
# Mimi / tokenizer / voice artifacts, then runs the emel converter to produce
# the enriched EMEL-lane GGUFs. Raw artifact paths feed the reference lane
# only; the *-emel.gguf outputs feed the EMEL lane.
#
# The reference driver build step activates with milestone 2 (mimi codec
# parity); until then --build-only is accepted and prints a notice.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="${EMEL_MOSHI_CPP_SOURCE_DIR:-$ROOT_DIR/build/moshi_cpp_reference/src}"
ARTIFACT_DIR="${EMEL_MOSHI_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/moshi_reference}"
MOSHI_CPP_REPO="${EMEL_MOSHI_CPP_REPO:-https://github.com/Codes4Fun/moshi.cpp.git}"
PIN_FILE="$ROOT_DIR/tools/bench/moshi_reference_ref.txt"
MOSHI_CPP_COMMIT="${EMEL_MOSHI_CPP_COMMIT:-$(tr -d '[:space:]' < "$PIN_FILE")}"
MOSHI_CPP_REF="${EMEL_MOSHI_CPP_REF:-$MOSHI_CPP_COMMIT}"

PERSONAPLEX_REVISION="1685c70257e525bc6c72470eee1ab2cacff3f709"
PERSONAPLEX_BASE="https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF/resolve/$PERSONAPLEX_REVISION"
MOSHI_COMMON_REVISION="a39d64307b971321140a67dc1bc4a8b0e43e4e6b"
MOSHI_COMMON_BASE="https://huggingface.co/Codes4Fun/moshi-common/resolve/$MOSHI_COMMON_REVISION"

MOSHI_MODEL="$ARTIFACT_DIR/model-q4_k.gguf"
MOSHI_MODEL_SHA256="73c4eb247740a48eebe5319ef767669dc8a06fd333b152a6cd6e12bacb64553d"
MOSHI_CONFIG="$ARTIFACT_DIR/personaplex-config.json"
MOSHI_CONFIG_SHA256="1b215765f6aafc6ef2592dadefcd8ad39c8b56d6eda25242be301b4af36b986a"
MIMI_MODEL="$ARTIFACT_DIR/mimi-e351c8d8-125.gguf"
MIMI_MODEL_SHA256="7e0c9ced83cbd035f70b82f1c5602673083fcccec006ea29f48d2e32c60ec697"
TOKENIZER_MODEL="$ARTIFACT_DIR/tokenizer_spm_32k_3.model"
TOKENIZER_MODEL_SHA256="78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d"
VOICE_MODEL="$ARTIFACT_DIR/voices/NATF0.gguf"
VOICE_MODEL_SHA256="af97a4dcf5e40f4c53801e964f7bce70051bc39c51545a596506243f8d83b8ea"
REFERENCE_AUDIO="$ARTIFACT_DIR/phase_moshi_440hz_24khz_mono.wav"

MOSHI_MODEL_EMEL="$ARTIFACT_DIR/model-q4_k-emel.gguf"
MIMI_MODEL_EMEL="$ARTIFACT_DIR/mimi-e351c8d8-125-emel.gguf"
VOICE_MODEL_EMEL="$ARTIFACT_DIR/voices/NATF0-emel.gguf"

FETCH_ONLY=false
BUILD_ONLY=false

usage() {
  cat <<'USAGE'
usage: scripts/setup_moshi_cpp_reference.sh [--fetch-only] [--build-only]

Fetches pinned moshi.cpp, downloads the sha256-pinned PersonaPlex-7B q4_k GGUF
plus Mimi/tokenizer/voice artifacts, generates the deterministic 24 kHz WAV
fixture, and converts the raw GGUFs to enriched EMEL-lane GGUFs via
tools/bench/moshi_gguf_convert.py.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch-only) FETCH_ONLY=true; shift ;;
    --build-only) BUILD_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

for tool in git python3 shasum curl; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if $BUILD_ONLY; then
  echo "note: the moshi.cpp reference driver build step activates with" \
       "milestone 2 (mimi codec parity); nothing to build yet"
  exit 0
fi

mkdir -p "$ARTIFACT_DIR/voices" "$(dirname "$REFERENCE_DIR")"

if [[ ! -d "$REFERENCE_DIR/.git" ]]; then
  git clone --filter=blob:none "$MOSHI_CPP_REPO" "$REFERENCE_DIR"
fi
git -C "$REFERENCE_DIR" fetch --tags --force origin "$MOSHI_CPP_REF"
git -C "$REFERENCE_DIR" checkout --detach "$MOSHI_CPP_REF"
resolved="$(git -C "$REFERENCE_DIR" rev-parse HEAD)"
if [[ "$resolved" != "$MOSHI_CPP_COMMIT" ]]; then
  echo "error: moshi.cpp ref resolved to $resolved, expected $MOSHI_CPP_COMMIT" >&2
  exit 1
fi

fetch_pinned() {
  local url="$1"
  local path="$2"
  local expected_sha="$3"
  if [[ ! -f "$path" ]]; then
    curl -fL "$url" -o "$path"
  fi
  local actual_sha
  actual_sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: sha256 mismatch for $path: $actual_sha != $expected_sha" >&2
    exit 1
  fi
}

fetch_pinned "$PERSONAPLEX_BASE/model-q4_k.gguf" "$MOSHI_MODEL" "$MOSHI_MODEL_SHA256"
fetch_pinned "$PERSONAPLEX_BASE/personaplex-config.json" "$MOSHI_CONFIG" "$MOSHI_CONFIG_SHA256"
fetch_pinned "$PERSONAPLEX_BASE/voices/NATF0.gguf" "$VOICE_MODEL" "$VOICE_MODEL_SHA256"
fetch_pinned "$MOSHI_COMMON_BASE/mimi-e351c8d8-125.gguf" "$MIMI_MODEL" "$MIMI_MODEL_SHA256"
fetch_pinned "$MOSHI_COMMON_BASE/tokenizer_spm_32k_3.model" "$TOKENIZER_MODEL" "$TOKENIZER_MODEL_SHA256"

if [[ ! -f "$REFERENCE_AUDIO" ]]; then
  python3 - "$REFERENCE_AUDIO" <<'PY'
import math
import struct
import sys
import wave

path = sys.argv[1]
sample_rate = 24000
with wave.open(path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    for index in range(sample_rate):
        value = 0.15 * math.sin(2.0 * math.pi * 440.0 * (index / sample_rate))
        wav.writeframesraw(struct.pack("<h", int(value * 32767)))
PY
fi

if $FETCH_ONLY; then
  exit 0
fi

convert_if_needed() {
  local source="$1"
  local output="$2"
  shift 2
  if [[ ! -f "$output" || "$source" -nt "$output" ]]; then
    python3 "$ROOT_DIR/tools/bench/moshi_gguf_convert.py" \
      --source "$source" --output "$output" \
      --manifest "${output%.gguf}.manifest.json" "$@"
  fi
}

convert_if_needed "$MOSHI_MODEL" "$MOSHI_MODEL_EMEL" \
  --config "$MOSHI_CONFIG" --tokenizer "$TOKENIZER_MODEL"
convert_if_needed "$MIMI_MODEL" "$MIMI_MODEL_EMEL" --config "$MOSHI_CONFIG"
convert_if_needed "$VOICE_MODEL" "$VOICE_MODEL_EMEL"

echo "EMEL_MOSHI_CPP_SOURCE=$REFERENCE_DIR"
echo "EMEL_MOSHI_REFERENCE_MODEL=$MOSHI_MODEL"
echo "EMEL_MOSHI_REFERENCE_MODEL_EMEL=$MOSHI_MODEL_EMEL"
echo "EMEL_MIMI_REFERENCE_MODEL=$MIMI_MODEL"
echo "EMEL_MIMI_REFERENCE_MODEL_EMEL=$MIMI_MODEL_EMEL"
echo "EMEL_MOSHI_TOKENIZER_MODEL=$TOKENIZER_MODEL"
echo "EMEL_MOSHI_VOICE_MODEL=$VOICE_MODEL"
echo "EMEL_MOSHI_VOICE_MODEL_EMEL=$VOICE_MODEL_EMEL"
echo "EMEL_MOSHI_CONFIG_JSON=$MOSHI_CONFIG"
echo "EMEL_MOSHI_REFERENCE_AUDIO=$REFERENCE_AUDIO"
