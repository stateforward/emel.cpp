#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/check_macos_binary_target.sh BINARY EXPECTED_MINIMUM" >&2
  exit 2
fi

binary="$1"
expected_minimum="$2"

if [[ ! -x "$binary" ]]; then
  echo "error: macOS target check requires an executable binary: $binary" >&2
  exit 1
fi
if ! command -v otool >/dev/null 2>&1; then
  echo "error: otool is required to inspect the macOS deployment target" >&2
  exit 1
fi

actual_minimum="$(
  otool -l "$binary" |
    awk '
      $1 == "cmd" && $2 == "LC_BUILD_VERSION" { in_build_version = 1; next }
      in_build_version && $1 == "minos" { print $2; exit }
    '
)"

if [[ -z "$actual_minimum" ]]; then
  echo "error: missing LC_BUILD_VERSION minimum OS in $binary" >&2
  exit 1
fi
if [[ "$actual_minimum" != "$expected_minimum" ]]; then
  echo "error: macOS minimum OS mismatch for $binary:" \
       "expected $expected_minimum, found $actual_minimum" >&2
  exit 1
fi

echo "macOS minimum OS verified: $binary minos=$actual_minimum"
