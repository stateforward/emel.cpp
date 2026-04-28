#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="${EMEL_WHISPER_CPP_SOURCE_DIR:-$ROOT_DIR/build/whisper_cpp_reference/src}"
BUILD_DIR="${EMEL_WHISPER_CPP_BUILD_DIR:-$ROOT_DIR/build/whisper_cpp_reference/build}"
ARTIFACT_DIR="${EMEL_WHISPER_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/whisper_reference}"
WHISPER_CPP_REPO="${EMEL_WHISPER_CPP_REPO:-https://github.com/ggml-org/whisper.cpp.git}"
WHISPER_CPP_REF="${EMEL_WHISPER_CPP_REF:-v1.7.6}"
WHISPER_CPP_COMMIT="${EMEL_WHISPER_CPP_COMMIT:-a8d002cfd879315632a579e73f0148d06959de36}"
REFERENCE_MODEL="$ARTIFACT_DIR/whisper-tiny-q8_0-whispercpp.gguf"
REFERENCE_MODEL_URL="${EMEL_WHISPER_REFERENCE_MODEL_URL:-https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/whisper.cpp/whisper-tiny-q8_0.gguf}"
REFERENCE_MODEL_SHA256="${EMEL_WHISPER_REFERENCE_MODEL_SHA256:-9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984}"
TOKENIZER_MODEL="${EMEL_WHISPER_TOKENIZER_MODEL:-$ROOT_DIR/tests/models/tokenizer-tiny.json}"
TOKENIZER_MODEL_URL="${EMEL_WHISPER_TOKENIZER_MODEL_URL:-https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/tokenizer-tiny.json}"
TOKENIZER_MODEL_SHA256="${EMEL_WHISPER_TOKENIZER_MODEL_SHA256:-dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759}"
REFERENCE_AUDIO="$ARTIFACT_DIR/phase99_440hz_16khz_mono.wav"
FETCH_ONLY=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/setup_whisper_cpp_reference.sh [--fetch-only] [--zig|--system]

Fetches and builds pinned whisper.cpp, downloads the pinned reference model, and
creates the deterministic Phase 99 WAV fixture.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) shift ;;
    --fetch-only) FETCH_ONLY=true; shift ;;
    --zig) USE_ZIG=true; shift ;;
    --system) USE_ZIG=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

for tool in git cmake python3 shasum curl; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

mkdir -p "$ARTIFACT_DIR" "$(dirname "$REFERENCE_DIR")"

if [[ ! -d "$REFERENCE_DIR/.git" ]]; then
  git clone --filter=blob:none "$WHISPER_CPP_REPO" "$REFERENCE_DIR"
fi
git -C "$REFERENCE_DIR" fetch --tags --force origin "$WHISPER_CPP_REF"
git -C "$REFERENCE_DIR" checkout --detach "$WHISPER_CPP_REF"
resolved="$(git -C "$REFERENCE_DIR" rev-parse HEAD)"
if [[ "$resolved" != "$WHISPER_CPP_COMMIT" ]]; then
  echo "error: whisper.cpp ref resolved to $resolved, expected $WHISPER_CPP_COMMIT" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE_MODEL" ]]; then
  curl -fL "$REFERENCE_MODEL_URL" -o "$REFERENCE_MODEL"
fi
actual_sha="$(shasum -a 256 "$REFERENCE_MODEL" | awk '{print $1}')"
if [[ "$actual_sha" != "$REFERENCE_MODEL_SHA256" ]]; then
  echo "error: reference model sha256 mismatch: $actual_sha != $REFERENCE_MODEL_SHA256" >&2
  exit 1
fi

mkdir -p "$(dirname "$TOKENIZER_MODEL")"
if [[ ! -f "$TOKENIZER_MODEL" ]]; then
  curl -fL "$TOKENIZER_MODEL_URL" -o "$TOKENIZER_MODEL"
fi
actual_tokenizer_sha="$(shasum -a 256 "$TOKENIZER_MODEL" | awk '{print $1}')"
if [[ "$actual_tokenizer_sha" != "$TOKENIZER_MODEL_SHA256" ]]; then
  echo "error: tokenizer sha256 mismatch: $actual_tokenizer_sha != $TOKENIZER_MODEL_SHA256" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE_AUDIO" ]]; then
  python3 - "$REFERENCE_AUDIO" <<'PY'
import math
import struct
import sys
import wave

path = sys.argv[1]
sample_rate = 16000
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

cmake_args=(
  -S "$REFERENCE_DIR"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DWHISPER_BUILD_TESTS=OFF
  -DWHISPER_BUILD_EXAMPLES=ON
  -DGGML_METAL=OFF
  -DWHISPER_COREML=OFF
  -DGGML_NATIVE=OFF
)
if $USE_ZIG; then
  if ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi
  cmake_args+=("-DCMAKE_C_COMPILER=$(command -v zig)")
  cmake_args+=("-DCMAKE_CXX_COMPILER=$(command -v zig)")
  cmake_args+=("-DCMAKE_ASM_COMPILER=$(command -v zig)")
  cmake_args+=("-DCMAKE_C_COMPILER_ARG1=cc")
  cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=c++")
  cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=cc")
fi
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel --target whisper-cli

cli_path="$(find "$BUILD_DIR" -type f -perm -111 -name 'whisper-cli' | head -1)"
if [[ -z "$cli_path" ]]; then
  echo "error: built whisper-cli not found under $BUILD_DIR" >&2
  exit 1
fi

echo "EMEL_WHISPER_CPP_CLI=$cli_path"
echo "EMEL_WHISPER_REFERENCE_MODEL=$REFERENCE_MODEL"
echo "EMEL_WHISPER_REFERENCE_AUDIO=$REFERENCE_AUDIO"
echo "EMEL_WHISPER_TOKENIZER_MODEL=$TOKENIZER_MODEL"
