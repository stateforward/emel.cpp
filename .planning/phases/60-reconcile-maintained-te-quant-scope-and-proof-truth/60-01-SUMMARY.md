---
phase: 60-reconcile-maintained-te-quant-scope-and-proof-truth
plan: 01
status: complete
completed: 2026-04-16
requirements-completed:
  - FIX-02
---

# Phase 60 Summary

## Outcome

Phase 60 is complete. The maintained TE quant-scope contract is truthful again: `v1.11` now
documents and proves exactly two approved TE fixtures, `TE-75M-q8_0.gguf` and
`TE-75M-q5_0.gguf`, while leaving other TE sibling artifacts unapproved. The maintained test and
bench workflow now reflects that scope directly, and q5 carries repo-owned proof instead of
init-only evidence.

## Delivered

- Updated `.planning/REQUIREMENTS.md` so `FIX-02` matches the live maintained TE scope instead of
  describing a q8-only world that no longer exists.
- Documented the q5 fixture in `tests/models/README.md` with its local size, checksum, and source
  URL, and tightened the out-of-scope language so arbitrary TE quant siblings are still excluded.
- Added explicit maintained-workflow fixture gating in `tests/embeddings/te_fixture_data.hpp` so
  the maintained env-switched bench/test lane only accepts the approved q8/q5 TE fixture names.
- Added maintained q5 proof and manifest coverage:
  - q5 upstream-golden proof in `tests/embeddings/te_proof_and_regression_tests.cpp`
  - q8/q5-only selector coverage in `tests/embeddings/text_embedding_lane_tests.cpp`
  - q5 README/size coverage in `tests/model/fixture_manifest_tests.cpp`

## Maintained q5 Evidence

- Focused q5 proof/selector/manifest tests now pass on the maintained coverage binary.
- Maintained q5 steady-state benchmark result (`EMEL_BENCH_ITERS=10`, `EMEL_BENCH_RUNS=5`):
  - Text: `6.655 ms/op` (`~150.3 embeddings/s`)
  - Audio: `23.228 ms/op` (`~43.1 embeddings/s`)
  - Image: `92.363 ms/op` (`~10.83 embeddings/s`)

## Requirement Close

- `FIX-02` is now satisfied because the maintained TE workflow no longer mixes a q8-only policy
  with silent q5 acceptance. The repo truth is explicit:
  - approved maintained TE fixtures: q8 and q5
  - unapproved TE siblings: still out of scope

## Notes

- This phase intentionally did not broaden the TE runtime claim beyond the two approved fixture
  variants. `q4_0` and any other sibling quantization remain unapproved until they gain their own
  maintained proof and benchmark coverage.
