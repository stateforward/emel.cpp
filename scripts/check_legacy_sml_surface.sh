#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: required tool missing: rg" >&2
  exit 2
fi

targets=(
  AGENTS.md
  CMakeLists.txt
  cmake
  docs
  include
  scripts
  src
  tests
  tools
)

patterns=(
  '#[[:space:]]*include[[:space:]]*[<"](boost/sml|sml\.hpp|boost/sml\.hpp)'
  '\bboost::sml\b'
  'using[[:space:]]+namespace[[:space:]]+boost::sml'
  '\bEMEL_BOOST_SML_[A-Z0-9_]*\b'
)

status=0

for pattern in "${patterns[@]}"; do
  if output=$(cd "$ROOT_DIR" && rg -n --color never \
      --glob '!docs/third_party/sml.md' \
      --glob '!tools/bench/logits_sml_vs_raw_fix.md' \
      --glob '!scripts/check_legacy_sml_surface.sh' \
      -- "$pattern" "${targets[@]}" 2>/dev/null); then
    echo "BLOCK: legacy SML surface reference"
    echo "$output"
    echo
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo "Legacy SML surface scan passed"
fi

exit "$status"
