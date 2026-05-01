---
phase: 99-whispercpp-parity-lane
plan: 01
requirements-completed: [PAR-01, PAR-02, PAR-03]
completed: 2026-04-26
---

# Phase 99 Plan 01: whisper.cpp Parity Lane - Execution Summary

**Phase Goal:** Add isolated `whisper.cpp` and EMEL parity lanes with stored comparison records for
the pinned fixture/audio pair.

**Status:** Complete for Phase 99 scope.

## Outcomes

### Pinned Reference Setup

- Added `scripts/setup_whisper_cpp_reference.sh`.
- The setup script clones `https://github.com/ggml-org/whisper.cpp.git`, checks out tag `v1.7.6`,
  and verifies commit `a8d002cfd879315632a579e73f0148d06959de36`.
- It downloads the pinned reference model from
  `oxide-lab/whisper-tiny-GGUF` commit `94468a6c81edab8c594d9b1d06ea1dfb64292327` and verifies
  SHA256 `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`.
- It generates `phase99_440hz_16khz_mono.wav` deterministically under `build/whisper_reference/`.
- The reference CLI builds under `build/whisper_cpp_reference/build/bin/whisper-cli`.

### Isolated Compare Lanes

- Added `tools/bench/whisper_emel_parity_runner.cpp`.
- The EMEL lane loads `tests/models/model-tiny-q80.gguf`, parses GGUF through EMEL loader events,
  builds the EMEL Whisper execution contract, drives `emel::whisper::encoder::sm`, then drives
  `emel::whisper::decoder::sm`.
- The reference lane invokes the pinned `whisper-cli` as an external process and does not share
  model, tokenizer, runtime, cache, or output objects with EMEL.
- Added `tools/bench/whisper_compare.py` and `scripts/bench_whisper_compare.sh` to produce raw
  `whisper_compare/v1` JSONL records and `build/whisper_compare/summary.json`.

### Quality Gate Wiring

- Added `tools/bench/reference_backends/whisper_cpp_asr.json` to identify the reference backend.
- Updated `tools/bench/CMakeLists.txt` so the Phase 99 runner can be built in the benchmark tools
  tree without widening the generic benchmark suite.
- Updated `scripts/quality_gates.sh` so
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` runs the wrapper directly.

## Verification Commands

- `scripts/setup_whisper_cpp_reference.sh --zig` - passed; pinned `whisper.cpp` built.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` - passed;
  summary verdict `bounded_drift` because EMEL currently publishes `token:50257` while
  `whisper.cpp` publishes `[Bell]`.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 99 files>" EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare scripts/quality_gates.sh` -
  passed.

## Stored Evidence

- Summary: `build/whisper_compare/summary.json`
- Raw EMEL lane: `build/whisper_compare/raw/emel.jsonl`
- Raw reference lane: `build/whisper_compare/raw/reference.jsonl`
- EMEL output: `build/whisper_compare/outputs/emel/transcript.txt`
- Reference output: `build/whisper_compare/outputs/reference/transcript.txt`

## Notes

- The Phase 99 verdict is `bounded_drift`, not an operational failure. Both lanes returned
  `status=ok`; drift is expected because the maintained EMEL runtime still publishes a deterministic
  token-id transcript while the full reference CLI publishes text.
- Timing and single-thread benchmark metadata are intentionally deferred to Phase 100.
