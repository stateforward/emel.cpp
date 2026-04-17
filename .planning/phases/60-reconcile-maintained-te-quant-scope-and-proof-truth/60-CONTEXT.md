---
phase: 60
slug: reconcile-maintained-te-quant-scope-and-proof-truth
created: 2026-04-16
status: ready
---

# Phase 60 Context

## Phase Boundary

Phase 60 exists because the live repo no longer matches the original q8-only TE slice story.
`TE-75M-q5_0.gguf` now initializes and benchmarks successfully, and the edge-performance work is
using it as the real deployment target. This phase makes that truthful instead of pretending the
runtime still rejects q5. It stays narrow: align the maintained TE quant scope across
requirements, fixture docs, maintained proof/benchmark surfaces, and the maintained test/bench
workflow. It does not widen to arbitrary TE quant siblings.

## Implementation Decisions

### Scope
- Promote `TE-75M-q5_0.gguf` as the second approved maintained TE fixture alongside
  `TE-75M-q8_0.gguf`.
- Keep `q8_0` as the default maintained fixture for the existing test/bench path.
- Make the maintained workflow explicit that only `q8_0` and `q5_0` are approved here.
- Add real maintained proof for q5 using the same upstream TE goldens, not ad hoc local scripts.

### Constraints
- No broad `omniembed` or arbitrary TE quant support claims.
- No hidden runtime behavior changes in the embedding state machine.
- Keep proof on the real `embeddings/generator` request path.
- Preserve the current maintained benchmark surface and its env-based fixture switch.

## Existing Code Insights

- `tests/embeddings/text_embedding_lane_tests.cpp` already proves `TE-75M-q5_0.gguf`
  initializes successfully.
- `tools/bench/embedding_generator_bench.cpp` already measures q5 through `EMEL_TE_FIXTURE`.
- `tests/embeddings/te_proof_and_regression_tests.cpp` already holds the upstream-golden proof
  harness for the default TE fixture.
- `tests/models/README.md` and `.planning/REQUIREMENTS.md` are the remaining contradictory docs:
  they still talk like q8 is the only maintained TE slice.

## Specific Ideas

- Update `FIX-02` to describe the real maintained TE quant scope: approved `q8_0` and `q5_0`
  fixtures only.
- Document q5 fixture provenance in `tests/models/README.md`.
- Add a dedicated q5 golden-proof test and fixture-manifest coverage.
- Gate the maintained fixture selector so test/bench workflows only accept the approved q8/q5
  paths.

## Deferred Ideas

- Supporting `q4_0` or any other TE sibling quantization
- Broad runtime quant-family negotiation beyond the approved maintained fixtures
- New public APIs or wider model-family claims
