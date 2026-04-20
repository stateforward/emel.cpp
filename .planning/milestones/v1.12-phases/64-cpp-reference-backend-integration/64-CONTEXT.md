---
phase: 64
slug: cpp-reference-backend-integration
created: 2026-04-17
status: complete
---

# Phase 64 Context

## Phase Boundary

Phase 64 moves the existing C++ reference lane onto the same manifest-driven compare architecture
used by the new Python backend. It keeps all reference setup in `tools/bench` and `scripts/`
without letting `src/` runtime code depend on reference-engine state.

## Implementation Decisions

### Scope
- Put the existing Liquid/mtmd reference lane behind a backend manifest.
- Keep the asset-default environment and build contract inside the existing shell wrapper.
- Make the compare driver invoke the manifest backend cleanly in JSONL mode.

### Constraints
- No reference-lane state may leak into `src/` runtime code.
- The manifest-driven compare workflow must use the real maintained wrapper so asset defaults stay
  truthful.

## Existing Code Insights

- The existing `embedding_reference_bench_runner` already had the approved ARM comparison matrix.
- The wrapper script `bench_embedding_reference_liquid.sh` already owned the asset and build
  contract; the compare architecture needed to respect that ownership instead of bypassing it.
