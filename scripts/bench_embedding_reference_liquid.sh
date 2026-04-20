#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"
BUILD_DIR="${EMEL_REFERENCE_BUILD_DIR:-$ROOT_DIR/build/bench_tools_liquid_ninja}"
REFERENCE_REPOSITORY="${EMEL_REFERENCE_REPOSITORY:-https://github.com/Liquid4All/benchmarks-llama.cpp.git}"
REFERENCE_REF="${EMEL_REFERENCE_REF:-0f345f3a1924ef274678a65bdec4b17f14e8fe9a}"
ASSET_DIR="${EMEL_REFERENCE_ASSET_DIR:-$ROOT_DIR/tests/models/reference}"
BUILD_ONLY=false
RUN_ONLY=false
DOWNLOAD_KNOWN_ASSETS=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_embedding_reference_liquid.sh [--build-only] [--run-only] [--download-known-assets] [--zig|--system]

Configures a dedicated Liquid AI multimodal reference benchmark build and builds
`embedding_reference_bench_runner`. The same runner can benchmark the approved
text and multimodal baseline matrix when the corresponding model assets are
configured.

Environment:
  EMEL_REFERENCE_BUILD_DIR   override build directory
  EMEL_REFERENCE_REPOSITORY  override Liquid reference repository
  EMEL_REFERENCE_REF         override Liquid reference commit/ref
  EMEL_REFERENCE_ASSET_DIR   local directory for downloaded baseline assets
  EMEL_REFERENCE_THREADS     runtime thread count for all reference cases
  EMEL_REFERENCE_TEXT_MODEL_ARCTIC_S
                             runtime Arctic S GGUF path
  EMEL_REFERENCE_TEXT_MODEL_EMBEDDINGGEMMA_300M
                             runtime EmbeddingGemma GGUF path
  EMEL_REFERENCE_VISION_MODEL
                             runtime LFM2-VL text model path
  EMEL_REFERENCE_VISION_MMPROJ
                             runtime LFM2-VL projector path
  EMEL_REFERENCE_AUDIO_MODEL runtime Ultravox text model path
  EMEL_REFERENCE_AUDIO_MMPROJ
                             runtime Ultravox projector path
  EMEL_REFERENCE_MM_MODEL    legacy vision-model fallback path
  EMEL_REFERENCE_MM_MMPROJ   legacy vision-mmproj fallback path
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=true ;;
    --run-only) RUN_ONLY=true ;;
    --download-known-assets) DOWNLOAD_KNOWN_ASSETS=true ;;
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

ensure_hf_tool() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    command -v huggingface-cli
    return 0
  fi
  echo "error: huggingface-cli not found (required for --download-known-assets)" >&2
  exit 1
}

download_hf_asset() {
  local repo="$1"
  local file="$2"
  local hf_bin="$3"
  "$hf_bin" download "$repo" "$file" \
    --local-dir "$ASSET_DIR" \
    --local-dir-use-symlinks False
}

mkdir -p "$ASSET_DIR"

default_arctic_model="$ASSET_DIR/snowflake-arctic-embed-s-Q8_0.gguf"
default_gemma_model="$ASSET_DIR/embeddinggemma-300M-Q8_0.gguf"
default_vision_model="$ASSET_DIR/LFM2-VL-450M-Q8_0.gguf"
default_vision_mmproj="$ASSET_DIR/mmproj-LFM2-VL-450M-Q8_0.gguf"
default_audio_model="$ASSET_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf"
default_audio_mmproj="$ASSET_DIR/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"

: "${EMEL_REFERENCE_TEXT_MODEL_ARCTIC_S:=$default_arctic_model}"
: "${EMEL_REFERENCE_TEXT_MODEL_EMBEDDINGGEMMA_300M:=$default_gemma_model}"
: "${EMEL_REFERENCE_VISION_MODEL:=$default_vision_model}"
: "${EMEL_REFERENCE_VISION_MMPROJ:=$default_vision_mmproj}"
: "${EMEL_REFERENCE_AUDIO_MODEL:=$default_audio_model}"
: "${EMEL_REFERENCE_AUDIO_MMPROJ:=$default_audio_mmproj}"
export EMEL_REFERENCE_TEXT_MODEL_ARCTIC_S
export EMEL_REFERENCE_TEXT_MODEL_EMBEDDINGGEMMA_300M
export EMEL_REFERENCE_VISION_MODEL
export EMEL_REFERENCE_VISION_MMPROJ
export EMEL_REFERENCE_AUDIO_MODEL
export EMEL_REFERENCE_AUDIO_MMPROJ

if $DOWNLOAD_KNOWN_ASSETS; then
  hf_bin="$(ensure_hf_tool)"
  download_hf_asset "yixuan-chia/snowflake-arctic-embed-s-GGUF" "snowflake-arctic-embed-s-Q8_0.gguf" "$hf_bin"
  download_hf_asset "ggml-org/embeddinggemma-300M-GGUF" "embeddinggemma-300M-Q8_0.gguf" "$hf_bin"
  download_hf_asset "ggml-org/LFM2-VL-450M-GGUF" "LFM2-VL-450M-Q8_0.gguf" "$hf_bin"
  download_hf_asset "ggml-org/LFM2-VL-450M-GGUF" "mmproj-LFM2-VL-450M-Q8_0.gguf" "$hf_bin"
  download_hf_asset "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF" "Llama-3.2-1B-Instruct-Q8_0.gguf" "$hf_bin"
  download_hf_asset "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF" "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf" "$hf_bin"
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

cmake_args=(
  -S "$TOOLS_DIR"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DEMEL_ENABLE_TESTS=OFF
  -DREF_IMPL_REPOSITORY="$REFERENCE_REPOSITORY"
  -DREF_IMPL_REF="$REFERENCE_REF"
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

if ! $RUN_ONLY; then
  cmake "${cmake_args[@]}"
  cmake --build "$BUILD_DIR" --parallel --target embedding_reference_bench_runner
fi

if $BUILD_ONLY; then
  exit 0
fi

"$BUILD_DIR/embedding_reference_bench_runner"
