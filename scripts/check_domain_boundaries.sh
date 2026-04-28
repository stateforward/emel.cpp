#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

status=0

check_no_matches() {
  local label="$1"
  local pattern="$2"
  shift 2

  if rg -n "$pattern" "$@" >/tmp/emel_domain_boundary_matches.$$ 2>/dev/null; then
    echo "error: domain boundary leak: ${label}" >&2
    cat /tmp/emel_domain_boundary_matches.$$ >&2
    status=1
  fi
  rm -f /tmp/emel_domain_boundary_matches.$$
}

cd "$ROOT_DIR"

check_no_matches "forbidden model-family runtime roots" \
  'emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper|model/whisper/(runtime|inference|encoder|decoder)|model::whisper::(runtime|inference|encoder|decoder)|speech/asr/whisper|speech::asr::whisper|speech/whisper|speech::whisper|recognizer/detail/whisper|recognizer::detail::whisper' \
  src tests tools CMakeLists.txt

check_no_matches "Whisper leaked into generic speech recognizer" \
  'whisper|model/whisper|speech/tokenizer/whisper|speech/encoder/whisper|speech/decoder/whisper|model::whisper|tokenizer::whisper|encoder::whisper|decoder::whisper' \
  src/emel/speech/recognizer tests/speech/recognizer

check_no_matches "Whisper model binding leaked into speech encoder/decoder runtime" \
  'emel/model/whisper|model::whisper' \
  src/emel/speech/encoder src/emel/speech/decoder

exit "$status"
