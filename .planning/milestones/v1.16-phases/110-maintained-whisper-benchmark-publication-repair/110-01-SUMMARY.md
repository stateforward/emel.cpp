---
phase: 110
plan: 1
status: complete
completed: 2026-04-27
requirements:
  - CLOSE-01
---

# Phase 110 Summary

## Completed Work

- `scripts/bench_whisper_single_thread.sh` now defaults `EMEL_WHISPER_EMEL_MODEL` to the pinned
  Phase 99 source model path instead of `tests/models/model-tiny-q80.gguf`.
- `tools/bench/whisper_benchmark.py` now records lane `model_sha256` in the summary and exits
  nonzero for `model_mismatch` or `transcript_mismatch`.
- The single-thread benchmark reference lane now uses the deterministic compare-lane policy flags:
  `--audio-ctx 50`, `--beam-size 1`, `--best-of 1`, and `--no-fallback`.
- `tools/bench/whisper_benchmark_tests.cpp` covers model mismatch, transcript mismatch, and
  deterministic reference-policy flag forwarding, plus per-iteration mismatch, warmup error, and
  missing reference-transcript failures.

## Evidence

- `build/whisper_compare_tools/whisper_benchmark_tests` passed with 6 test cases and 86
  assertions.
- The repaired single-thread wrapper passed with one measured iteration and wrote
  `build/whisper_benchmark/benchmark_summary.json` with both lanes on model SHA
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`, transcript `[C]`, and
  `status: ok`.
