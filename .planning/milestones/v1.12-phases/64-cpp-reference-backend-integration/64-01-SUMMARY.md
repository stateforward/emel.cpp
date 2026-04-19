---
phase: 64-cpp-reference-backend-integration
plan: 01
status: complete
completed: 2026-04-17
requirements-completed:
  - CPP-01
  - CPP-02
---

# Phase 64 Summary

## Outcome

Phase 64 is complete. The maintained C++ reference lane now runs as a pluggable manifest backend
under the shared compare contract instead of as a one-off benchmark entrypoint.

## Delivered

- Finished the JSONL contract cleanup in `embedding_reference_bench_runner`.
- Added `--run-only` to `scripts/bench_embedding_reference_liquid.sh` so manifest-driven runs can
  reuse the maintained asset-default wrapper without mixing build chatter into JSONL output.
- Updated `tools/bench/reference_backends/liquid_cpp.json` so the compare driver invokes the real
  wrapper contract for the C++ backend.

## C++ Backend Truth

- The C++ backend still owns its own model assets, build, and runtime setup.
- The compare driver now treats it exactly like a pluggable backend instead of special-casing a
  raw binary path.
- The compare summary truthfully marks the Liquid text lane as `unavailable` for similarity
  because it is a baseline backend, not a parity backend.
