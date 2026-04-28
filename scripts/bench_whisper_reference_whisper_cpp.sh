#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  EMEL_WHISPER_CPP_CLI=/path/to/whisper-cli \
  EMEL_WHISPER_REFERENCE_MODEL=/path/to/whisper.cpp/whisper-tiny-q4_k.gguf \
  EMEL_WHISPER_REFERENCE_AUDIO=/path/to/audio.wav \
    scripts/bench_whisper_reference_whisper_cpp.sh

Runs the isolated whisper.cpp reference lane for the pinned Whisper tiny artifact.
Reference objects produced here must not feed EMEL-owned runtime state.
EOF
  exit 0
fi

if [[ -z "${EMEL_WHISPER_CPP_CLI:-}" ]]; then
  echo "EMEL_WHISPER_CPP_CLI is required and must point to whisper-cli" >&2
  exit 2
fi

if [[ -z "${EMEL_WHISPER_REFERENCE_MODEL:-}" ]]; then
  echo "EMEL_WHISPER_REFERENCE_MODEL is required and must point to the reference model" >&2
  exit 2
fi

if [[ -z "${EMEL_WHISPER_REFERENCE_AUDIO:-}" ]]; then
  echo "EMEL_WHISPER_REFERENCE_AUDIO is required and must point to a WAV fixture" >&2
  exit 2
fi

if [[ ! -x "${EMEL_WHISPER_CPP_CLI}" ]]; then
  echo "whisper.cpp CLI is not executable: ${EMEL_WHISPER_CPP_CLI}" >&2
  exit 2
fi

if [[ ! -f "${EMEL_WHISPER_REFERENCE_MODEL}" ]]; then
  echo "reference model is missing: ${EMEL_WHISPER_REFERENCE_MODEL}" >&2
  exit 2
fi

if [[ ! -f "${EMEL_WHISPER_REFERENCE_AUDIO}" ]]; then
  echo "reference audio fixture is missing: ${EMEL_WHISPER_REFERENCE_AUDIO}" >&2
  exit 2
fi

threads="${EMEL_WHISPER_REFERENCE_THREADS:-4}"

echo "# whisper_reference_backend=whisper.cpp"
echo "# whisper_reference_model=${EMEL_WHISPER_REFERENCE_MODEL}"
echo "# whisper_reference_audio=${EMEL_WHISPER_REFERENCE_AUDIO}"
echo "# whisper_reference_threads=${threads}"

exec "${EMEL_WHISPER_CPP_CLI}" \
  --model "${EMEL_WHISPER_REFERENCE_MODEL}" \
  --file "${EMEL_WHISPER_REFERENCE_AUDIO}" \
  --threads "${threads}"
