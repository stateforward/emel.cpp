# Phase 156-01 Summary: Parity Dependency Manifest Gate Closure

## Outcome

Phase 156 closed `MANIFEST-01` and `MANIFEST-02` by moving dependency-manifest emission and
freshness decisions onto maintained paritychecker and quality-gate entrypoints.

## Changes

- Added paritychecker CLI operations:
  - `--write-dependency-manifest <path>`
  - `--check-dependency-manifest <path>`
  - `--dependency-manifest-uncertain`
- Added `dependency_manifest::inspect(...)` so production CLI checks classify missing, stale, and
  uncertain manifest freshness and reuse `requires_full_gate(...)`.
- Added the maintained generated baseline at `tools/paritychecker/dependency_manifest.txt`.
- Wired `scripts/quality_gates.sh` to emit/check the manifest before auto parity skips; missing,
  stale, or uncertain freshness forces `scripts/paritychecker.sh`.
- Updated manifest documentation with production CLI and quality-gate freshness examples.
- Added focused tests for CLI emission, fresh/stale/missing/uncertain checks, baseline freshness,
  and quality-gate source wiring.

## Regression Reproduction

The new tests failed before implementation because:

- `paritychecker --write-dependency-manifest` and `--check-dependency-manifest` returned usage
  failures.
- `tools/paritychecker/dependency_manifest.txt` did not exist.
- `scripts/quality_gates.sh` did not reference the manifest CLI or uncertain freshness override.

After the implementation, those tests passed.

## Verification

Commands passed:

```sh
cmake --build build/paritychecker_zig --target paritychecker_tests paritychecker -j2
ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests
build/paritychecker_zig/paritychecker --check-dependency-manifest \
  tools/paritychecker/dependency_manifest.txt
env EMEL_QUALITY_GATES_CHANGED_FILES='tools/paritychecker/parity_runner.cpp:tools/paritychecker/parity_dependency_manifest.cpp:tools/paritychecker/parity_dependency_manifest.hpp:tools/paritychecker/paritychecker_tests.cpp:tools/paritychecker/dependency_manifest.md:tools/paritychecker/dependency_manifest.txt:scripts/quality_gates.sh:.planning/ROADMAP.md:.planning/REQUIREMENTS.md:.planning/PROJECT.md:.planning/STATE.md:.planning/milestones/v1.18-ROADMAP.md:.planning/milestones/v1.18-REQUIREMENTS.md:.planning/phases/156-parity-dependency-manifest-gate-closure/156-CONTEXT.md:.planning/phases/156-parity-dependency-manifest-gate-closure/156-01-PLAN.md' \
  scripts/quality_gates.sh
```

The scoped quality gate passed and the generated timing snapshot churn was restored before commit.
