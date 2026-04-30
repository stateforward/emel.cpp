---
phase: 111
status: fixed
reviewed: 2026-04-27
---

# Phase 111 Code Review

## Findings Fixed

| Finding | Result | Fix |
|---------|--------|-----|
| Benchmark could pass despite earlier iteration mismatch | fixed | `tools/bench/whisper_benchmark.py` now compares measured records by `iteration_index` and records `first_mismatch`. |
| Recognizer retained initialize payload views in persistent context | fixed | `recognize` carries model/tokenizer for the top-level dispatch and `recognize_ctx` carries same-RTC contract/tokenizer handoff data. |
| Warmup lane failures were ignored | fixed | Warmup records are included in the final failure scan. |
| Reference success did not require transcript output | fixed | Missing reference transcript output is recorded as `missing_transcript` error. |

## Residual Note

The benchmark still accepts an overridden reference CLI. The maintained scripts build or locate the
pinned `whisper.cpp` v1.7.6 CLI by default; overriding `EMEL_WHISPER_CPP_CLI` remains an explicit
operator-controlled escape hatch and is not used by closeout evidence.
