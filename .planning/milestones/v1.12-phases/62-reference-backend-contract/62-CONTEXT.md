---
phase: 62
slug: reference-backend-contract
created: 2026-04-17
status: complete
---

# Phase 62 Context

## Phase Boundary

Phase 62 defines the shared contract and lane-isolation rules for the new pluggable embedding
comparison architecture. It stays in `tools/bench` and planning artifacts; it does not widen
`src/` runtime scope or let reference-engine state cross into the EMEL lane.

## Implementation Decisions

### Scope
- Introduce one canonical `embedding_compare/v1` record format for both EMEL and reference lanes.
- Make the maintained EMEL embedding runner emit that contract without changing the actor-owned
  embedding path.
- Make the C++ reference runner emit the same contract cleanly in JSONL mode.
- Keep reference backend selection outside the EMEL lane through manifest-driven tooling.

### Constraints
- Preserve strict EMEL/reference lane isolation.
- Keep emitted records machine-readable with binary vector dumps for later similarity comparison.
- Avoid polluting JSONL mode with metadata chatter.

## Existing Code Insights

- The maintained EMEL lane already had stable timing and output-anchor metadata in
  `embedding_generator_bench.cpp`; the missing piece was a shared machine-readable contract.
- The existing C++ reference runner already exposed enough output/timing detail, but its output was
  still human-oriented and mixed metadata with results.

## Deferred Ideas

- Remote or service-hosted reference backends
- Generation-surface compare unification
