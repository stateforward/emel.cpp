---
phase: 99-whispercpp-parity-lane
verified: 2026-04-26T15:59:16Z
status: passed
score: 4/4 must-haves verified
---

# Phase 99: whisper.cpp Parity Lane Verification Report

**Phase Goal:** Add isolated `whisper.cpp` and EMEL parity lanes with stored comparison records for
the pinned fixture/audio pair.
**Verified:** 2026-04-26T15:59:16Z
**Status:** passed

## Goal Achievement

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Reference lane invokes `whisper.cpp` without sharing reference state with EMEL. | VERIFIED | `scripts/setup_whisper_cpp_reference.sh` builds pinned `whisper.cpp` under `build/`; `tools/bench/whisper_compare.py` invokes `whisper-cli` as a separate process. |
| 2 | EMEL lane loads, runs, and publishes output only through EMEL-owned code. | VERIFIED | `tools/bench/whisper_emel_parity_runner.cpp` uses EMEL GGUF loader, Whisper contract, encoder actor, and decoder actor; no reference includes or objects are used. |
| 3 | Stored records include backend identity, model checksum, audio fixture identity, transcript, metadata, and verdict. | VERIFIED | `build/whisper_compare/summary.json` contains `whisper_compare_summary/v1`; raw lane records include backend IDs, SHA256s, audio fixture ID, transcript, token/timestamp metadata, checksums, and status. |
| 4 | Wrapper hard-fails missing tools or fixtures instead of skipping proof. | VERIFIED | Setup and compare scripts use `set -euo pipefail`, required-tool checks, checksum verification, executable/file validation, and non-zero returns for lane errors. |

**Score:** 4/4 must-haves verified

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| PAR-01 | SATISFIED | Pinned `whisper.cpp` v1.7.6 commit `a8d002cfd879315632a579e73f0148d06959de36` built and invoked for the same Phase 99 audio fixture. |
| PAR-02 | SATISFIED | EMEL lane is a separate runner using EMEL loader/runtime actors and the staged EMEL model fixture only. |
| PAR-03 | SATISFIED | `build/whisper_compare/summary.json` and raw JSONL records store transcript, checksum, fixture/model identity, metadata, and verdict. |

## Automated Checks

- `scripts/setup_whisper_cpp_reference.sh --zig` - passed.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` - passed with
  `status=bounded_drift reason=transcript_mismatch`.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 99 files>" EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare scripts/quality_gates.sh` -
  passed.

## Human Verification Required

None.

## Residual Notes

- The stored parity verdict is `bounded_drift` because EMEL emits `token:50257` and the reference
  lane emits `[Bell]`. Both lanes are operationally successful.
- Phase 100 must add single-thread CPU timing metadata; Phase 99 intentionally records parity
  identity and output drift only.
