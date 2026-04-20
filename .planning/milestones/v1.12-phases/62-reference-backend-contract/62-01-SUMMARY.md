---
phase: 62-reference-backend-contract
plan: 01
status: complete
completed: 2026-04-17
requirements-completed:
  - REF-01
  - REF-02
  - ISO-01
---

# Phase 62 Summary

## Outcome

Phase 62 is complete. The maintained embedding compare architecture now has one canonical
`embedding_compare/v1` contract shared across EMEL and reference lanes, and the contract stays
outside `src/` runtime ownership so backend selection does not alter the EMEL lane.

## Delivered

- Added `tools/bench/embedding_compare_contract.hpp` with canonical result fields for timing,
  backend identity, fixture identity, output anchors, optional binary vector dumps, and explicit
  error fields.
- Updated `embedding_generator_bench_runner` so the maintained EMEL lane can emit JSONL compare
  records without losing the existing human-readable benchmark snapshot mode.
- Updated `embedding_reference_bench_runner` so JSONL mode stays machine-readable and does not mix
  metadata lines into the output stream.
- Documented the backend-manifest contract in `tools/bench/reference_backends/README.md`.

## Contract Truth

- Backend selection now lives in manifest/tooling space rather than in the EMEL lane.
- EMEL and reference lanes emit the same schema with separate lane identities and separate output
  files.
- The EMEL lane still runs only through `embedding_generator_bench_runner` and the maintained
  `embeddings/generator` actor path.
