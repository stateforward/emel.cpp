#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/build_with_zig.sh"
"$ROOT_DIR/scripts/test_with_coverage.sh"
"$ROOT_DIR/scripts/lint_snapshot.sh"
"$ROOT_DIR/scripts/bench.sh" --snapshot
