#!/usr/bin/env bash
# Fetches and prepares the PersonaPlex-MLX Mimi reference lane: clones the
# pinned personaplex-mlx checkout (tools/bench/personaplex_mlx_ref.txt),
# builds its Python 3.12/3.13 venv, downloads the sha256-pinned Mimi
# safetensors weights plus the raw/enriched Mimi GGUFs for the EMEL lane,
# and writes the driver shim consumed by scripts/bench_mimi_compare.sh
# --reference=personaplex-mlx.
#
# MLX requires an Apple Silicon macOS host; this script hard-fails anywhere
# else per the missing-tools rule.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${EMEL_PERSONAPLEX_MLX_ARTIFACT_DIR:-$ROOT_DIR/build/personaplex_mlx_reference}"
MOSHI_ARTIFACT_DIR="${EMEL_MOSHI_REFERENCE_ARTIFACT_DIR:-$ROOT_DIR/build/moshi_reference}"
SRC_DIR="$ARTIFACT_DIR/src"
VENV_DIR="$ARTIFACT_DIR/venv"
DRIVER_SHIM="$ARTIFACT_DIR/mimi_driver"

PERSONAPLEX_MLX_REPOSITORY="https://github.com/mu-hashmi/personaplex-mlx"
PERSONAPLEX_MLX_REF_FILE="$ROOT_DIR/tools/bench/personaplex_mlx_ref.txt"

# Mimi weights: Kyutai's ungated MLX distribution of the same
# e351c8d8-checkpoint125 codec the moshi.cpp lane's GGUF was converted from,
# so both reference lanes share weight provenance.
MIMI_SAFETENSORS_REVISION="a31ebad976783a3ea356d6beb8b8e826d2d45a68"
MIMI_SAFETENSORS_URL="https://huggingface.co/kyutai/moshiko-mlx-bf16/resolve/$MIMI_SAFETENSORS_REVISION/tokenizer-e351c8d8-checkpoint125.safetensors"
MIMI_SAFETENSORS="$ARTIFACT_DIR/tokenizer-e351c8d8-checkpoint125.safetensors"
MIMI_SAFETENSORS_SHA256="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"

# EMEL-lane Mimi artifacts; keep these pins in sync with
# scripts/setup_moshi_cpp_reference.sh (same mirror, same revisions).
PERSONAPLEX_REVISION="1685c70257e525bc6c72470eee1ab2cacff3f709"
PERSONAPLEX_BASE="https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF/resolve/$PERSONAPLEX_REVISION"
MOSHI_COMMON_REVISION="a39d64307b971321140a67dc1bc4a8b0e43e4e6b"
MOSHI_COMMON_BASE="https://huggingface.co/Codes4Fun/moshi-common/resolve/$MOSHI_COMMON_REVISION"
MOSHI_CONFIG="$MOSHI_ARTIFACT_DIR/personaplex-config.json"
MOSHI_CONFIG_SHA256="1b215765f6aafc6ef2592dadefcd8ad39c8b56d6eda25242be301b4af36b986a"
MIMI_MODEL="$MOSHI_ARTIFACT_DIR/mimi-e351c8d8-125.gguf"
MIMI_MODEL_SHA256="7e0c9ced83cbd035f70b82f1c5602673083fcccec006ea29f48d2e32c60ec697"
MIMI_MODEL_EMEL="$MOSHI_ARTIFACT_DIR/mimi-e351c8d8-125-emel.gguf"

for tool in git curl shasum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: personaplex-mlx reference lane needs an Apple Silicon macOS host" >&2
  exit 1
fi

PYTHON_BIN=""
for candidate in python3.12 python3.13; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "error: required tool missing: python3.12 or python3.13 (mlx wheels)" >&2
  exit 1
fi

if [[ ! -f "$PERSONAPLEX_MLX_REF_FILE" ]]; then
  echo "error: missing personaplex-mlx pin at $PERSONAPLEX_MLX_REF_FILE" >&2
  exit 1
fi
PERSONAPLEX_MLX_REF="$(head -n 1 "$PERSONAPLEX_MLX_REF_FILE" | tr -d '[:space:]')"
if [[ -z "$PERSONAPLEX_MLX_REF" ]]; then
  echo "error: empty personaplex-mlx pin in $PERSONAPLEX_MLX_REF_FILE" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$MOSHI_ARTIFACT_DIR"

fetch_pinned() {
  local url="$1"
  local path="$2"
  local expected_sha="$3"
  local actual_sha
  if [[ -f "$path" ]]; then
    actual_sha="$(shasum -a 256 "$path" | awk '{print $1}')"
    if [[ "$actual_sha" == "$expected_sha" ]]; then
      return 0
    fi
    echo "warning: refetching $path (sha256 mismatch)" >&2
  fi
  curl -fL --retry 3 "$url" -o "$path"
  actual_sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: sha256 mismatch for $path: $actual_sha != $expected_sha" >&2
    exit 1
  fi
}

if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone "$PERSONAPLEX_MLX_REPOSITORY" "$SRC_DIR"
fi
if ! git -C "$SRC_DIR" rev-parse --verify --quiet "$PERSONAPLEX_MLX_REF^{commit}" >/dev/null; then
  git -C "$SRC_DIR" fetch origin "$PERSONAPLEX_MLX_REF"
fi
git -C "$SRC_DIR" checkout --quiet "$PERSONAPLEX_MLX_REF"
resolved_ref="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [[ "$resolved_ref" != "$PERSONAPLEX_MLX_REF" ]]; then
  echo "error: personaplex-mlx resolved to '$resolved_ref', expected pinned '$PERSONAPLEX_MLX_REF'" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -e "$SRC_DIR"
"$VENV_DIR/bin/python" -c "from personaplex_mlx import models; models.mimi.mimi_202407(16)" \
  || { echo "error: personaplex_mlx import check failed" >&2; exit 1; }

fetch_pinned "$MIMI_SAFETENSORS_URL" "$MIMI_SAFETENSORS" "$MIMI_SAFETENSORS_SHA256"
fetch_pinned "$MOSHI_COMMON_BASE/mimi-e351c8d8-125.gguf" "$MIMI_MODEL" "$MIMI_MODEL_SHA256"
fetch_pinned "$PERSONAPLEX_BASE/personaplex-config.json" "$MOSHI_CONFIG" "$MOSHI_CONFIG_SHA256"

# The converter embeds the config as hparams metadata, so a refreshed config
# must force a reconvert, and a converter-only patch must too: the scoped
# speech_codec_mimi_mlx gate otherwise compares against a stale enriched
# artifact and can pass without exercising the changed converter.
CONVERTER="$ROOT_DIR/tools/bench/moshi_gguf_convert.py"
if [[ ! -f "$MIMI_MODEL_EMEL" || "$MIMI_MODEL" -nt "$MIMI_MODEL_EMEL" ||
      "$MOSHI_CONFIG" -nt "$MIMI_MODEL_EMEL" ||
      "$CONVERTER" -nt "$MIMI_MODEL_EMEL" ]]; then
  # Use the interpreter discovered by the version check above: hosts that only
  # ship python3.12/python3.13 (no python3 shim) must convert with it too.
  "$PYTHON_BIN" "$ROOT_DIR/tools/bench/moshi_gguf_convert.py" \
    --source "$MIMI_MODEL" --output "$MIMI_MODEL_EMEL" \
    --manifest "${MIMI_MODEL_EMEL%.gguf}.manifest.json" \
    --config "$MOSHI_CONFIG"
fi

cat > "$DRIVER_SHIM" <<SHIM
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" "$ROOT_DIR/tools/bench/speech/personaplex_mlx_mimi_driver.py" "\$@"
SHIM
chmod +x "$DRIVER_SHIM"

echo "personaplex-mlx reference lane ready:"
echo "  src:     $SRC_DIR @ $resolved_ref"
echo "  weights: $MIMI_SAFETENSORS"
echo "  driver:  $DRIVER_SHIM"
