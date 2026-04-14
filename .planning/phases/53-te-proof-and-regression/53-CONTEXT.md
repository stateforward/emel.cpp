---
phase: 53
slug: te-proof-and-regression
created: 2026-04-14
status: ready
---

# Phase 53 Context

## Phase Boundary

Phase 53 closes the maintained `v1.11` TE slice by proving it against stored upstream golden
embeddings and by adding small cross-modal smoke coverage that lives inside the normal repo test
and gate flow. The scope is proof and regression hardening only: keep the maintained anchors
`red-square` and `pure-tone-440hz`, prove text/image/audio behavior against stored upstream truth,
and keep parity with existing tokenizer/parity fixtures while the gates stay green. This phase does
not widen into public API work, broader semantic evaluation, or additional TE quant variants.

## Implementation Decisions

### Proof Surface
- Keep proof on the maintained `tests/models/TE-75M-q8_0.gguf` slice and the canonical in-memory
  anchors already defined under `tests/embeddings/fixtures/te75m/`.
- Load stored upstream golden vectors from repo-owned fixture files and compare them directly inside
  EMEL tests rather than relying on manual notebooks or ad hoc scripts.
- Keep the smoke contract narrow: prove one small set of cross-modal relations instead of claiming
  broad retrieval quality.

### Regression Hardening
- Integrate the TE golden proof into the normal doctest and `scripts/quality_gates.sh` path so the
  maintained slice fails loudly on regressions.
- Preserve existing BERT GGUF paritychecker behavior while landing TE proof work. The maintained TE
  vocab and the existing parity fixture do not store WordPiece pieces the same way, so any WPM
  compatibility fix must keep both truthful.

### Scope Guardrails
- Do not expand into generic media decode, vector search, batching, or public API work.
- Do not weaken proof by lowering expectations to ad hoc manual inspection or one-off scripts.
- Keep the first maintained smoke relations deterministic and tiny.

## Existing Code Insights

### Reusable Assets
- `tests/embeddings/te_fixture.hpp` already owns the maintained TE fixture loader and canonical
  payload synthesis helpers.
- `tests/embeddings/fixtures/te75m/README.md` already documents the narrow proof corpus.
- `src/emel/embeddings/generator/` already owns the shared trimodal embedding publication path.
- `tools/paritychecker/` and `tests/models/bert-base-uncased-q4_k_m.gguf` already pin an existing
  WordPiece regression surface that must keep passing while TE proof lands.

### Specific Risks
- The maintained `mdbr-leaf-ir` vocab uses raw starts plus `##` continuations, while the existing
  BERT GGUF parity fixture stores word-start pieces with `▁` markers. A TE-only tokenizer fix
  would regress paritychecker.
- The proof must remain cheap enough to run under the normal gate flow while still being anchored
  to stored upstream truth.

## Specific Ideas

- Add one dedicated `tests/embeddings/te_proof_and_regression_tests.cpp` file that:
  - caches canonical text/image/audio TE outputs once
  - compares them against stored upstream golden vectors
  - checks tiny cross-modal smoke relations for the canonical anchors
- Keep the stored goldens in `tests/embeddings/fixtures/te75m/` with the generator provenance
  documented alongside them.
- Extend WPM coverage so the maintained TE proof and the existing BERT paritychecker both remain
  truthful.

## Deferred Ideas

- broader semantic evaluation or retrieval benchmarking
- additional TE quant slices
- public embedding API work
- batching or async embedding requests

## Primary Sources

- `tests/embeddings/te_fixture.hpp`
- `tests/embeddings/te_proof_and_regression_tests.cpp`
- `tests/embeddings/fixtures/te75m/README.md`
- `src/emel/text/encoders/wpm/detail.hpp`
- `tests/text/encoders/wpm_tests.cpp`
- `tools/paritychecker/CMakeLists.txt`

---
*Phase: 53-te-proof-and-regression*
*Context gathered: 2026-04-14*
