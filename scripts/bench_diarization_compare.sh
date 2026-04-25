#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"
BUILD_DIR="${EMEL_BENCH_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
OUTPUT_DIR="${EMEL_DIARIZATION_COMPARE_OUTPUT_DIR:-$ROOT_DIR/build/diarization_compare}"
COMPARE_PYTHON="${EMEL_DIARIZATION_COMPARE_PYTHON:-}"
ONNX_REFERENCE_MODEL="${EMEL_DIARIZATION_ONNX_REFERENCE_MODEL:-}"
ONNX_REFERENCE_FEATURES="${EMEL_DIARIZATION_ONNX_REFERENCE_FEATURES:-}"
ONNX_REFERENCE_EXPECTED_SHA256="${EMEL_DIARIZATION_ONNX_REFERENCE_MODEL_SHA256:-5df5e883c8dae4e0ecba77739f3db38997c2ae57153de2583d625afb6abb2be0}"
PYTORCH_REFERENCE_MODEL="${EMEL_DIARIZATION_PYTORCH_REFERENCE_MODEL:-}"
PYTORCH_REFERENCE_AUDIO="${EMEL_DIARIZATION_PYTORCH_REFERENCE_AUDIO:-}"
PYTORCH_REFERENCE_PYTHON="${EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON:-}"
PYTORCH_REFERENCE_DEVICE="${EMEL_DIARIZATION_PYTORCH_REFERENCE_DEVICE:-cpu}"
PYTORCH_REFERENCE_VENV="${EMEL_DIARIZATION_PYTORCH_REFERENCE_VENV:-$ROOT_DIR/build/diarization_pytorch_ref_venv}"
SETUP_PYTORCH_REFERENCE_ENV=false
SKIP_EMEL_BUILD=false
USE_ZIG=true

usage() {
  cat <<'USAGE'
usage: scripts/bench_diarization_compare.sh [--output-dir DIR] [--onnx-reference-model FILE]
                                            [--onnx-reference-features FILE]
                                            [--pytorch-reference-model MODEL_OR_NEMO]
                                            [--pytorch-reference-audio WAV]
                                            [--setup-pytorch-reference-env]
                                            [--skip-emel-build] [--zig|--system]

Builds the maintained EMEL bench runner, then runs the deterministic diarization compare workflow
against the recorded maintained baseline lane. When an ONNX reference model is supplied, the
workflow also exports the maintained EMEL feature tensor and compares an ONNX Runtime reference
lane without substituting the recorded baseline. When a PyTorch/NeMo model is supplied, the
workflow also runs the documented NeMo Sortformer diarize path on the maintained WAV fixture and
reports it as a separate reference backend.

Environment:
  EMEL_BENCH_BUILD_DIR                override EMEL bench build directory
  EMEL_DIARIZATION_COMPARE_OUTPUT_DIR override compare output directory
  EMEL_DIARIZATION_COMPARE_PYTHON     optional Python executable for compare + ONNX runner
  EMEL_DIARIZATION_ONNX_REFERENCE_MODEL optional ONNX reference model path
  EMEL_DIARIZATION_ONNX_REFERENCE_FEATURES optional precomputed feature tensor path
  EMEL_DIARIZATION_ONNX_REFERENCE_MODEL_SHA256 expected ONNX file SHA-256
  EMEL_DIARIZATION_PYTORCH_REFERENCE_MODEL optional HF model id or local .nemo path
  EMEL_DIARIZATION_PYTORCH_REFERENCE_AUDIO optional WAV fixture path
  EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON optional Python executable with torch+nemo
  EMEL_DIARIZATION_PYTORCH_REFERENCE_VENV optional uv venv path
  EMEL_DIARIZATION_PYTORCH_REFERENCE_DEVICE optional torch device, default cpu

Reference ONNX model:
  build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx
Reference PyTorch model:
  nvidia/diar_streaming_sortformer_4spk-v2.1

Lane roles:
  recorded.diarization.baseline       self-recorded regression snapshot
  pytorch.nemo.sortformer.v2_1        parity reference lane
  onnx.sortformer.v2_1                benchmark reference lane, cross-checked against PyTorch
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --onnx-reference-model)
      ONNX_REFERENCE_MODEL="${2:-}"
      shift 2
      ;;
    --onnx-reference-features)
      ONNX_REFERENCE_FEATURES="${2:-}"
      shift 2
      ;;
    --pytorch-reference-model)
      PYTORCH_REFERENCE_MODEL="${2:-}"
      shift 2
      ;;
    --pytorch-reference-audio)
      PYTORCH_REFERENCE_AUDIO="${2:-}"
      shift 2
      ;;
    --pytorch-reference-python)
      PYTORCH_REFERENCE_PYTHON="${2:-}"
      shift 2
      ;;
    --pytorch-reference-device)
      PYTORCH_REFERENCE_DEVICE="${2:-}"
      shift 2
      ;;
    --setup-pytorch-reference-env)
      SETUP_PYTORCH_REFERENCE_ENV=true
      shift
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
  "$ROOT_DIR/tools/bench/diarization_compare.py" \
  --emel-runner "$BUILD_DIR/bench_runner" \
  --output-dir "$OUTPUT_DIR"
)
if [[ -z "$COMPARE_PYTHON" ]]; then
  if [[ -n "$ONNX_REFERENCE_MODEL" && -x "$ROOT_DIR/build/onnx_ref_venv/bin/python" ]]; then
    COMPARE_PYTHON="$ROOT_DIR/build/onnx_ref_venv/bin/python"
  else
    COMPARE_PYTHON="python3"
  fi
fi
if [[ -n "$ONNX_REFERENCE_MODEL" ]]; then
  compare_args+=(--onnx-reference-model "$ONNX_REFERENCE_MODEL")
  compare_args+=(--onnx-reference-expected-sha256 "$ONNX_REFERENCE_EXPECTED_SHA256")
fi
if [[ -n "$ONNX_REFERENCE_FEATURES" ]]; then
  compare_args+=(--onnx-reference-features "$ONNX_REFERENCE_FEATURES")
fi
if [[ -n "$PYTORCH_REFERENCE_MODEL" ]]; then
  if $SETUP_PYTORCH_REFERENCE_ENV; then
    "$ROOT_DIR/scripts/setup_diarization_pytorch_ref_env.sh" --venv "$PYTORCH_REFERENCE_VENV"
  fi
  if [[ -z "$PYTORCH_REFERENCE_PYTHON" ]]; then
    if [[ -x "$PYTORCH_REFERENCE_VENV/bin/python" ]]; then
      PYTORCH_REFERENCE_PYTHON="$PYTORCH_REFERENCE_VENV/bin/python"
    else
      PYTORCH_REFERENCE_PYTHON="python3"
    fi
  fi
  compare_args+=(--pytorch-reference-model "$PYTORCH_REFERENCE_MODEL")
  compare_args+=(--pytorch-reference-python "$PYTORCH_REFERENCE_PYTHON")
  compare_args+=(--pytorch-reference-device "$PYTORCH_REFERENCE_DEVICE")
  if [[ -n "$PYTORCH_REFERENCE_AUDIO" ]]; then
    compare_args+=(--pytorch-reference-audio "$PYTORCH_REFERENCE_AUDIO")
  fi
fi

"$COMPARE_PYTHON" "${compare_args[@]}"
