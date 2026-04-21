---
phase: 71
slug: maintained-reference-backend-integration
created: 2026-04-20
status: in_progress
---

# Phase 71 Context

## Phase Boundary

Phase 71 brings the first maintained non-EMEL generative backend onto the shared
`generation_compare/v1` surface. The scope stays in `tools/bench`, wrapper scripts, backend
manifests, tests, and planning artifacts; it does not widen `src/` runtime ownership or leak
reference-backend setup into the EMEL generation lane.

## Implementation Decisions

### Scope
- Reuse the existing maintained C++ llama.cpp generation lane as the first pluggable generation
  reference backend instead of inventing a new runtime.
- Add a manifest-driven backend selector and wrapper surface for generation, mirroring the proven
  embedding-side shape where it helps.
- Introduce a repo-owned generation compare driver that runs EMEL and the selected reference
  backend independently and stores explicit machine-readable error records on failures.

### Constraints
- Keep backend-specific build/run setup in scripts and backend manifests only.
- Preserve strict EMEL/reference lane isolation and keep EMEL on the existing `bench_runner`
  generation surface.
- Surface backend build/run failures as reproducible `generation_compare/v1` error records instead
  of crashing or silently dropping compare output.
- Defer operator-facing publication verdict categories and closeout proof packaging to later
  phases.

## Existing Code Insights

- `tools/bench/embedding_compare.py` already provides the closest existing manifest-driven compare
  driver pattern, including backend build/run hooks and deterministic error-record insertion.
- `tools/bench/generation_bench.cpp` already emits `generation_compare/v1` JSONL from both the
  EMEL lane and the built-in C++ reference lane.
- `scripts/bench.sh` and the current generation compare path still treat the reference lane as a
  built-in benchmark mode rather than an explicitly selected backend manifest.

## Deferred Ideas

- One operator-facing wrapper script that publishes compare artifacts and verdicts
- Explicit exact-match / bounded-drift / non-comparable publication categories
- Milestone closeout proof bundles and traceability refresh
