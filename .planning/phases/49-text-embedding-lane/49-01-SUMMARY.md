---
phase: 49-text-embedding-lane
plan: 01
status: complete
completed: 2026-04-14
requirements-completed:
  - TXT-01
  - TXT-02
---

# Phase 49 Summary

## Outcome

Phase 49 is complete. EMEL now has a maintained repo-owned TE text embedding lane that runs
through `src/emel/embeddings/generator/`, binds through the existing text conditioner/tokenizer
seam, executes the real TE text tower on `tests/models/TE-75M-q8_0.gguf`, and publishes normalized
shared-space embeddings with supported Matryoshka truncation.

## Delivered

- Added the embedding-session actor surface under `src/emel/embeddings/generator/` with explicit
  initialize, text-embed, callback, and error channels.
- Implemented the maintained TE text runtime path:
  `LEAF-IR BertModel -> mean pool -> dense(384 -> 768) -> normalize -> TE text projection head ->
  1280-d shared embedding`.
- Added maintained text-lane tests that prove:
  - normalized full-dimension output
  - supported truncation at `768/512/256/128`
  - explicit invalid truncation rejection
  - init/embed callback publication
  - init/conditioning error-path coverage
  - low-level embedding helper coverage needed to keep repo gates green
- Pinned `tests/models/mdbr-leaf-ir-vocab.txt` as the maintained WordPiece tokenizer truth for the
  text lane because the TE GGUF fixture omits tokenizer metadata needed for the maintained slice.
- Fixed an unrelated `token_batcher` signed-overflow crash that surfaced during the required
  repo-wide gate run.

## Validation

- `TXT-01` validated: maintained text input now returns a normalized `1280`-dimensional TE
  embedding.
- `TXT-02` validated: maintained text requests now support `768/512/256/128` truncation with
  renormalization and explicit unsupported-dimension rejection.

## Gate Result

- `scripts/quality_gates.sh` passed.
- Global line coverage recovered to `90.2%`.
- Benchmark regressions remained warning-only and did not fail the gate script.
