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

machine_toc() {
  local headers=()
  while IFS= read -r h; do
    headers+=("$h")
  done < <(find "$ROOT_DIR/src/emel" -type f -name 'sm.hpp' \
    ! -path "$ROOT_DIR/src/emel/sm.hpp" | sort)
  for h in "${headers[@]}"; do
    local rel_emel="${h#"$ROOT_DIR/src/emel/"}"
    local dir="${rel_emel%/sm.hpp}"
    local name="${dir//\//_}"
    echo "- \`docs/architecture/$name.md\`"
  done
}

generate_readme() {
  local toc_file="$TMP_DIR/machine_toc.txt"
  machine_toc > "$toc_file"
  sed -e "/__MACHINE_TOC__/r $toc_file" -e "/__MACHINE_TOC__/d" <<'MD'
# EMEL

Deterministic, production-grade C++ inference engine built around Boost.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

This inference engine is being implemented by AI under human engineering and architecture direction.

## Why EMEL

EMEL exists to make inference behavior explicit and verifiable. Instead of ad-hoc control flow,
orchestration is modeled as Boost.SML state machines with deterministic, testable transitions.
That enables:

1. Clear operational semantics and failure modes.
2. Deterministic, reproducible inference paths.
3. High-performance, C-compatible boundaries without dynamic dispatch in hot paths.
4. Auditable parity work against reference implementations without copying their control flow.

## The name

“EMEL” is pronounced like “ML”. It’s a short, neutral name that doesn’t carry existing
assumptions or baggage. It’s intentionally low-ceremony while we iterate on the core design.

## Build and test

```bash
scripts/build_with_zig.sh
scripts/test_with_coverage.sh
scripts/lint_snapshot.sh
```

### Why Zig for builds

Zig’s C/C++ toolchain gives us consistent, fast, cross-platform builds without forcing a full
dependency on any single system compiler or SDK. It keeps the default dev path reproducible,
while still allowing native toolchains when needed.

### Why CMake for tests and coverage

Coverage and CI tooling are already standardized around CMake + CTest + llvm-cov/gcovr in this
repo. Using CMake for test/coverage builds keeps gates deterministic and portable across CI
environments, while Zig remains the default for day-to-day builds.

## Documentation

- `docs/architecture/` (generated state-machine diagrams)
- `docs/sml.md` (Boost.SML conventions and usage)
- `docs/gaps.md` (parity audit status)

## State machine reference

__MACHINE_TOC__

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
