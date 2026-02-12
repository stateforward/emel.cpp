#!/usr/bin/env bash
set -euo pipefail

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: required tool missing: clang-format" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "error: required tool missing: git" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$ROOT_DIR/snapshots/lint/clang_format.txt"
CURRENT="$(mktemp)"
trap 'rm -f "$CURRENT"' EXIT

while IFS= read -r f; do
  if ! diff -u "$f" <(clang-format "$f") >/dev/null; then
    rel="${f#"$ROOT_DIR/"}"
    echo "$rel" >> "$CURRENT"
  fi
done < <(find "$ROOT_DIR/src" "$ROOT_DIR/include" "$ROOT_DIR/tests" -type f \
  \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) | sort)

if [[ "${1:-}" == "--update" ]]; then
  cp "$CURRENT" "$BASELINE"
  echo "updated $BASELINE"
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: missing baseline $BASELINE (run scripts/lint_snapshot.sh --update)" >&2
  exit 1
fi

if ! diff -u "$BASELINE" "$CURRENT"; then
  echo "error: lint snapshot regression" >&2
  exit 1
fi
