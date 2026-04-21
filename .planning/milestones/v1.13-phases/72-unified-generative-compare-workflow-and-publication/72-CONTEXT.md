---
phase: 72
slug: unified-generative-compare-workflow-and-publication
created: 2026-04-20
status: in_progress
---

# Phase 72 Context

## Phase Boundary

Phase 72 turns the manifest-driven generation backend seam into one operator-facing compare
workflow with reproducible artifacts and truthful verdict labels. The scope stays in wrapper
scripts, compare-tool publication semantics, regression tests, docs, and planning artifacts.

## Implementation Decisions

### Scope
- Publish one wrapper entrypoint for EMEL-vs-reference generation compare runs.
- Extend compare-summary semantics to distinguish exact matches, bounded drift, and
  non-comparable workloads explicitly.
- Document the emitted artifact layout, backend identity, workload-manifest provenance, and
  verdict semantics.

### Constraints
- Keep benchmark publication (`scripts/bench.sh --compare`) distinct from the new operator-facing
  compare-artifact workflow.
- Preserve explicit backend identity and workload provenance in every published artifact.
- Treat single-lane workloads truthfully as `non_comparable` instead of hiding them.

## Existing Code Insights

- Phase 71 already added `tools/bench/generation_compare.py` and
  `scripts/bench_generation_compare.sh`, so the remaining work is mostly verdict semantics,
  end-user docs, and operator-facing publication polish.

## Deferred Ideas

- Broader maintained backend matrix beyond the first llama.cpp lane
- Remote/service-hosted generation compare backends
