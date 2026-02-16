#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mode="${1:-}"
if [[ -n "$mode" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 1
fi

README_FILE="$ROOT_DIR/README.md"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

generate_readme() {
  cat <<'MD'
# EMEL

Deterministic, production-grade C++ inference engine built around Boost.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

## Why EMEL

EMEL exists to make inference behavior explicit and verifiable. Instead of ad-hoc control flow,
orchestration is modeled as Boost.SML state machines with deterministic, testable transitions.
That enables:

1. Clear operational semantics and failure modes.
2. Deterministic, reproducible inference paths.
3. High-performance, C-compatible boundaries without dynamic dispatch in hot paths.
4. Auditable parity work against reference implementations without copying their control flow.

## Build and test

```bash
scripts/build_with_zig.sh
scripts/test_with_coverage.sh
scripts/lint_snapshot.sh
```

## Documentation

- `docs/architecture/` (generated state-machine diagrams)
- `docs/sml.md` (Boost.SML conventions and usage)
- `docs/gaps.md` (parity audit status)

## Regenerating docs

```bash
scripts/generate_docs.sh
```

Use `scripts/generate_docs.sh --check` in CI to validate generated artifacts.
MD
}

"$ROOT_DIR/scripts/generate_puml.sh" ${mode:+$mode}
"$ROOT_DIR/scripts/generate_md.sh" ${mode:+$mode}

if [[ "${mode:-}" == "--check" ]]; then
  README_TMP="$TMP_DIR/README.md"
  generate_readme > "$README_TMP"
  if [[ ! -f "$README_FILE" ]]; then
    echo "error: missing $README_FILE" >&2
    exit 1
  fi
  if ! diff -u "$README_FILE" "$README_TMP"; then
    echo "error: README.md out of sync" >&2
    exit 1
  fi
  exit 0
fi

generate_readme > "$README_FILE"
echo "updated: $README_FILE"
