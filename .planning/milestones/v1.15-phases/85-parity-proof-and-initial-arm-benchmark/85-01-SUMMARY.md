---
requirements-completed: []
---

# Phase 85 Summary: Parity Proof And Initial ARM Benchmark

## Completed

- Added a deterministic Sortformer diarization parity test for
  `sortformer_canonical_multispeaker_16khz_15s_v1`.
- Added the maintained `diarization_sortformer` benchmark suite with separated EMEL and fixed
  reference-baseline lanes.
- Extended benchmark compare output with `# diarization_sortformer:` metadata lines reporting lane,
  case, model, fixture, workload, output count, checksum, profile parameters, and proof status.
- Documented the supported Sortformer model, fixed input profile, output contract, proof fixture,
  and pre-optimization limitations through the benchmarks doc template.

## Notes

- Benchmark snapshots were not updated. The new benchmark row intentionally appears as a new entry
  until snapshot update approval is given.
- The reference lane is a trusted fixed segment baseline for the proof fixture, not llama.cpp
  execution and not a performance parity claim.
