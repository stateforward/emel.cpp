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

## Regenerating docs

```bash
scripts/generate_docs.sh
```

Use `scripts/generate_docs.sh --check` in CI to validate generated artifacts.
