#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${EMEL_DIARIZATION_PYTORCH_REFERENCE_VENV:-$ROOT_DIR/build/diarization_pytorch_ref_venv}"
PYTHON_BIN="${EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON_BIN:-python3.11}"
REQUIREMENTS_FILE="${EMEL_DIARIZATION_PYTORCH_REFERENCE_REQUIREMENTS:-$ROOT_DIR/tools/bench/diarization_pytorch_reference_requirements.txt}"
CHECK_IMPORT=true

usage() {
  cat <<'USAGE'
usage: scripts/setup_diarization_pytorch_ref_env.sh [--venv DIR] [--python PYTHON]
                                                   [--requirements FILE] [--no-check-import]

Creates or syncs the isolated PyTorch/NeMo Sortformer parity reference environment with uv.

Environment:
  EMEL_DIARIZATION_PYTORCH_REFERENCE_VENV          output venv directory
  EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON_BIN    Python executable, default python3.11
  EMEL_DIARIZATION_PYTORCH_REFERENCE_REQUIREMENTS  pinned requirements file

After setup, run the compare with:
  EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON=build/diarization_pytorch_ref_venv/bin/python \
    scripts/bench_diarization_compare.sh \
      --pytorch-reference-model nvidia/diar_streaming_sortformer_4spk-v2.1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="${2:-}"
      shift 2
      ;;
    --no-check-import)
      CHECK_IMPORT=false
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

if ! command -v uv >/dev/null 2>&1; then
  echo "error: required tool missing: uv" >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "error: requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

uv pip sync --python "$VENV_DIR/bin/python" "$REQUIREMENTS_FILE"

if $CHECK_IMPORT; then
  "$VENV_DIR/bin/python" - <<'PY'
import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel

print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"nemo_model_class={SortformerEncLabelModel.__name__}")
PY
fi

echo "EMEL_DIARIZATION_PYTORCH_REFERENCE_PYTHON=$VENV_DIR/bin/python"
