---
phase: 58
slug: embedding-generator-benchmark-publication
created: 2026-04-15
status: ready
---

# Phase 58 Context

## Phase Boundary

Phase 58 closes the remaining benchmark publication gap from the milestone audit. The TE embedding
runtime already has maintained correctness proof, but `tools/bench` still lacks a maintained path
that drives initialize plus a real embedding request through `src/emel/embeddings/generator`
instead of timing helper-local or adjacent generation surfaces.

## Implementation Decisions

### Benchmark Scope
- Publish one maintained TE benchmark path through the real embedding generator request flow.
- Keep the benchmark anchored to the shipped TE-75M `omniembed` slice and its maintained request
  contract.
- Prefer a dedicated maintained EMEL benchmark surface over forcing the existing compare runner into
  an unnatural helper-only or mismatched reference shape.

### Runner Shape
- Reuse the existing `tools/bench` fixture-loading and reporting patterns where they fit.
- Avoid modifying benchmark snapshot expectations unless the user explicitly asks for snapshot
  updates.
- Prefer a dedicated executable or dedicated suite entry when the existing `bench_runner
  --mode=compare` pairing rules would otherwise force a fake or ad hoc reference lane.

### Proof Shape
- Make the benchmark evidence auditable as embedding-generator coverage, not just benchmark code
  that reaches into helpers.
- Include the exact maintained TE fixture and request modality used for the benchmark case.
- Preserve the current benchmark split rule: EMEL-owned code on the EMEL lane only.

## Existing Code Insights

### Relevant Surfaces
- `tools/bench/CMakeLists.txt` currently wires `bench_runner` cases for generation, kernels,
  memory, logits, planner, tokenizer, and GBNF, but nothing for `emel/embeddings/generator`.
- `tools/bench/bench_main.cpp` assumes paired EMEL/reference suites for `--mode=compare`.
- `tools/bench/gbnf/bench_main.cpp` already shows the repo is willing to add a dedicated bench
  runner when a benchmark surface does not fit the main compare harness cleanly.

### Constraints
- AGENTS forbids presenting helper-local timing as maintained runtime evidence.
- AGENTS also forbids using llama.cpp objects on the EMEL lane, so any benchmark must drive the
  EMEL slice strictly through EMEL-owned setup and request paths.
- Phase 57 verification is still running the global benchmark compare gate, so benchmark code edits
  should wait until that process releases `tools/bench`.

## Specific Ideas

- Add a dedicated embedding benchmark source under `tools/bench` that initializes the maintained TE
  fixture once, then measures one real text/image/audio embedding request through
  `emel::embeddings::generator::sm`.
- Pair that source with a small dedicated executable, similar to `gbnf_bench_runner`, so the new
  benchmark can publish maintained EMEL-only evidence without disturbing compare-mode case counts.
- Reuse the existing fixture helpers from `tests/embeddings/te_fixture.hpp` or a benchmark-local
  equivalent if the current helper is test-only.

## Deferred Ideas

- A reference-lane embedding compare benchmark
- Snapshot or docs updates for benchmark output
- Additional modality variants beyond the one maintained TE publication case needed for the audit

---
*Phase: 58-embedding-generator-benchmark-publication*
*Context gathered: 2026-04-15*
