# EMEL

Deterministic, production-grade C++ inference engine with Boost.SML orchestration.
WARNING: Documentation is under active development and may change frequently.

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
