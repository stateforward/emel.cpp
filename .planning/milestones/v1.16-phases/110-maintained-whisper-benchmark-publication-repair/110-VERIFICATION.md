---
phase: 110
status: passed
verified: 2026-04-27
requirements:
  - CLOSE-01
---

# Phase 110 Verification

## Requirement Verification

| Requirement | Result | Evidence |
|-------------|--------|----------|
| CLOSE-01 | passed | Benchmark publication truth is repaired: same pinned source model SHA is recorded for both lanes, deterministic reference policy matches compare, and mismatches fail publication. |

## Source-Backed Checks

- `scripts/bench_whisper_single_thread.sh` sets `emel_model` from
  `EMEL_WHISPER_EMEL_MODEL:-$model_path`, where `model_path` is the pinned Phase 99 source model
  under `build/whisper_reference`.
- `tools/bench/whisper_benchmark.py` writes `model_sha256` into each lane summary and sets
  `status: error` for model SHA or transcript mismatch.
- The benchmark reference invocation now includes `--audio-ctx 50`, `--beam-size 1`,
  `--best-of 1`, and `--no-fallback`, matching the maintained compare lane.

## Command Evidence

```sh
build/whisper_compare_tools/whisper_benchmark_tests
```

Result: 6 test cases passed, 86 assertions passed.

```sh
EMEL_WHISPER_BENCH_WARMUPS=0 EMEL_WHISPER_BENCH_ITERATIONS=1 \
  scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build
```

Result: `benchmark_status=ok reason=ok`.

`build/whisper_benchmark/benchmark_summary.json` records:

- EMEL model SHA:
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`
- reference model SHA:
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`
- EMEL transcript: `[C]`
- reference transcript: `[C]`
- summary status: `ok`
