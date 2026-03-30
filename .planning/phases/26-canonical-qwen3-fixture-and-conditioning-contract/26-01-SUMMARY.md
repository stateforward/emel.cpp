# Phase 26-01 Summary

## Outcome

Cut the maintained generation fixture identity over to the canonical
`Qwen3-0.6B-Q8_0.gguf` slice and proved that the maintained paritychecker and bench surfaces now
resolve Qwen explicitly instead of silently reusing the old Llama anchor.

## Implementation

- Added the official Qwen3 fixture provenance entry to `tests/models/README.md`, including the
  approved size and SHA256.
- Updated maintained generation help text and fixture constants in `tools/paritychecker` and
  `tools/bench` so the only maintained v1.6 fixture anchor is
  `tests/models/Qwen3-0.6B-Q8_0.gguf`.
- Tightened maintained generation subprocess coverage in
  `tools/paritychecker/paritychecker_tests.cpp` so the canonical Qwen fixture is required and
  same-basename impostors outside `tests/models/` are rejected.

## Test Coverage

- `./build/paritychecker_zig/paritychecker_tests --test-case='*canonical generation fixture*' --no-breaks`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Qwen3-0.6B-Q8_0.gguf --text hello --max-tokens 1 --write-generation-baseline /tmp/qwen26-baseline.txt`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only`

## Deviations From Plan

- `[Rule 3 - Blocking]` The original 26-01 verification expected a successful maintained Qwen
  bench run, but the real maintained Qwen runtime is not brought up until Phase 27. The phase was
  re-scoped to require truthful pre-runtime failure on the canonical Qwen fixture instead of an
  impossible success-path benchmark proof.

## Result

- Maintained generation surfaces now point at the canonical Qwen fixture.
- The old Llama fixture anchor no longer survives on the maintained Qwen path.
- Before Phase 27 runtime bring-up, real paritychecker and bench commands fail explicitly on the
  canonical Qwen fixture instead of drifting back to the old Llama slice.
