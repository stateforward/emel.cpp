#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mode="${1:-}"
if [[ -n "$mode" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 1
fi

"$ROOT_DIR/scripts/generate_puml.sh" ${mode:+$mode}
"$ROOT_DIR/scripts/generate_md.sh" ${mode:+$mode}

