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

check_no_matches_except() {
  local label="$1"
  local pattern="$2"
  local allow_pattern="$3"
  shift 3

  if rg -n "$pattern" "$@" >/tmp/emel_domain_boundary_matches.$$ 2>/dev/null; then
    if grep -Ev "$allow_pattern" /tmp/emel_domain_boundary_matches.$$ \
        >/tmp/emel_domain_boundary_filtered.$$; then
      echo "error: domain boundary leak: ${label}" >&2
      cat /tmp/emel_domain_boundary_filtered.$$ >&2
      status=1
    fi
    rm -f /tmp/emel_domain_boundary_filtered.$$
  fi
  rm -f /tmp/emel_domain_boundary_matches.$$
}

cd "$ROOT_DIR"

check_no_matches "forbidden model-family runtime roots" \
  'emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper|model/whisper/(runtime|inference|encoder|decoder)|model::whisper::(runtime|inference|encoder|decoder)|speech/asr/whisper|speech::asr::whisper|speech/whisper|speech::whisper|recognizer/detail/whisper|recognizer::detail::whisper' \
  src tests tools CMakeLists.txt

check_no_matches_except "legacy text generator root" \
  'emel/generator|emel::generator|namespace emel::generator|src/emel/generator|tests/generator' \
  '^(src/emel/generator/|tests/text/generator/legacy_compatibility_tests\.cpp:[0-9]+:|scripts/(quality_gates|test_with_coverage)\.sh:[0-9]+:.*src/emel/generator)' \
  src tests tools scripts/quality_gates.sh scripts/test_with_coverage.sh CMakeLists.txt

check_no_matches "text generator actor internals in maintained generation parity/benchmark lanes" \
  'emel/text/generator/(detail|actions|guards)\.hpp|emel::text::generator::(detail|action|guard)::|emel::text::generator::prefill::guard::|generation_internal_diagnostics' \
  tools/bench/generation_bench.cpp tools/paritychecker/parity_runner.cpp tools/paritychecker/parity_runner.hpp

check_no_matches "Whisper leaked into generic speech recognizer" \
  'whisper|model/whisper|speech/tokenizer/whisper|speech/encoder/whisper|speech/decoder/whisper|model::whisper|tokenizer::whisper|encoder::whisper|decoder::whisper' \
  src/emel/speech/recognizer tests/speech/recognizer

check_no_matches "Whisper model binding leaked into speech encoder/decoder runtime" \
  'emel/model/whisper|model::whisper' \
  src/emel/speech/encoder src/emel/speech/decoder

exit "$status"
