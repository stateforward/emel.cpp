# Phase 244 Verification

status: passed

All commands were run from:
`/shared/stateforward/emel.cpp`

## Must-Have Verification

| Must-have | Evidence | Status |
|-----------|----------|--------|
| Maintained `kernel_x86_64` benchmark suite is wired | `tools/bench/bench_runner_registry.cpp`, `tools/bench/kernel/x86_64_bench.cpp`, and `scripts/quality_gates.sh` expose and select the suite | PASS |
| Optimized x86_64 flash and quantized paths are benchmarked | `tools/bench/kernel/x86_64_bench.cpp` publishes `op_flash_attn_ext_decode_like`, `op_mul_mat_q2_k_q8_k`, `op_mul_mat_q3_k_q8_k`, and `op_mul_mat_q6_k_q8_k` entries that abort if the optimized actor counters do not advance | PASS |
| Benchmark preflight is suite-scoped and non-mutating | `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` ran and did not update snapshots | PASS |
| Publication baseline updated | `scripts/bench.sh --snapshot --update --suite=kernel_x86_64` merged 19 `kernel/x86_64/*` entries into `snapshots/bench/benchmarks.txt` | PASS |
| Generation publication baselines updated | paritychecker wrote the maintained LFM2 `10`, `100`, and `1000` token baselines under `snapshots/parity/` | PASS |
| Unsupported feature claims remain excluded | Phase 243 unsupported x86 flag scan found no AVX-512, AVX-VNNI, AMX, BF16, or native FP16 compile-flag claims | PASS |
| Required quality gate | changed-file scoped quality gate passed with `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64` | PASS |

## Evidence Summary

- `node .codex/get-shit-done/bin/gsd-tools.cjs init phase-op 244` reports
  context and plan present.
- `node .codex/get-shit-done/bin/gsd-tools.cjs roadmap analyze` reports Phase
  244 as planned with one plan.
- `git diff --check` passes.
- Initial `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64`
  configured and built the suite-scoped benchmark runner, then failed only
  because the maintained benchmark baseline lacked the first 15 common
  `kernel/x86_64/*` entries:
  - `kernel/x86_64/op_sqrt`
  - `kernel/x86_64/op_div`
  - `kernel/x86_64/op_sin`
  - `kernel/x86_64/op_unary_neg`
  - `kernel/x86_64/op_unary_relu`
  - `kernel/x86_64/op_mul`
  - `kernel/x86_64/op_mul_mat`
  - `kernel/x86_64/op_sub`
  - `kernel/x86_64/op_add`
  - `kernel/x86_64/op_soft_max`
  - `kernel/x86_64/op_dup`
  - `kernel/x86_64/op_cos`
  - `kernel/x86_64/op_sqr`
  - `kernel/x86_64/op_unary_exp`
  - `kernel/x86_64/op_log`
- The preflight also prints `error: no benchmark entries matched selected suite`
  because the selected suite has no existing baseline entries to compare yet.
  That is a publication-baseline absence, not a runtime execution failure.
- Source-backed milestone audit found an `XBN-01` gap: those first 15 common
  entries did not prove the x86_64 optimized flash and q2/q3/q6 benchmark
  lanes. `tools/bench/kernel/x86_64_bench.cpp` was repaired so the maintained
  `kernel_x86_64` suite now also publishes counter-checked optimized-path
  entries for:
  - `kernel/x86_64/op_flash_attn_ext_decode_like`
  - `kernel/x86_64/op_mul_mat_q2_k_q8_k`
  - `kernel/x86_64/op_mul_mat_q3_k_q8_k`
  - `kernel/x86_64/op_mul_mat_q6_k_q8_k`
- Direct benchmark smoke passed:
  `EMEL_BENCH_SUITE=kernel_x86_64 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=emel`
  emitted 19 `kernel/x86_64/*` entries, including the optimized flash and
  q2/q3/q6 entries.
- Direct benchmark compare smoke passed with the same 19 EMEL/reference rows:
  `EMEL_BENCH_SUITE=kernel_x86_64 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=compare`.
- Final `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` passed
  after the approved snapshot update and includes the optimized flash and
  q2/q3/q6 entries.
- Published benchmark entry names are:
  `op_add`, `op_cos`, `op_div`, `op_dup`, `op_log`, `op_mul`, `op_mul_mat`,
  `op_mul_mat_q2_k_q8_k`, `op_mul_mat_q3_k_q8_k`,
  `op_mul_mat_q6_k_q8_k`, `op_flash_attn_ext_decode_like`, `op_sin`,
  `op_soft_max`, `op_sqr`, `op_sqrt`, `op_sub`, `op_unary_exp`,
  `op_unary_neg`, and `op_unary_relu`.
- Temp maintained generation baseline candidate writes succeeded without touching
  `snapshots/parity/`:
  - `--max-tokens=10`: status 0, generated 10 tokens, output 20 bytes,
    optimized flash 228, optimized q6 390, shared q6 0.
  - `--max-tokens=100`: status 0, generated 100 tokens, output 248 bytes,
    optimized flash 768, optimized q6 1380, shared q6 0.
  - `--max-tokens=1000`: status 0, generated 1000 tokens, output 2498 bytes,
    optimized flash 6168, optimized q6 11280, shared q6 0.
- Candidate-vs-snapshot diffs show the checked-in stale baselines have
  `trace_token_count=0` while candidates include the live token IDs and score
  gaps. Output lengths change from `22 -> 20`, `277 -> 248`, and `2866 -> 2498`
  for `10`, `100`, and `1000` tokens respectively.

## Final Verification

User approved snapshot updates. After the source-backed audit exposed the
missing optimized benchmark entries, `tools/bench/kernel/x86_64_bench.cpp` was
repaired and `scripts/bench.sh --snapshot --update --suite=kernel_x86_64`
updated `snapshots/bench/benchmarks.txt`. Paritychecker updated the maintained
LFM2 `10`, `100`, and `1000` token publication baselines, the focused parity
publication test passed, `scripts/bench.sh --snapshot --compare
--suite=kernel_x86_64` passed, and the changed-file scoped quality gate passed
with `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64`.

Phase 244 is fully verified for `XBN-01` and `XBN-02`.
