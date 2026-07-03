---
phase: 244
slug: benchmark-attribution-and-publication-truth
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-25
---

# Phase 244 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | benchmark snapshot gate, paritychecker, source scans, quality gate |
| Config file | `scripts/bench.sh`; `scripts/quality_gates.sh` |
| Quick run command | `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` |
| Gate command | `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64 scripts/quality_gates.sh` |
| Current gate status | passed after approved benchmark and generation publication snapshot updates |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 244-01-01 | XBN-01, XBN-02 | benchmark preflight | `scripts/bench.sh --snapshot --compare --suite=kernel_x86_64` | green |
| 244-01-02 | XBN-01, XBN-02 | approved snapshot writes | `scripts/bench.sh --snapshot --update --suite=kernel_x86_64` plus paritychecker baseline writes | green |
| 244-01-03 | XBN-01 | source-backed audit gap repair | `tools/bench/kernel/x86_64_bench.cpp`; benchmark smoke and snapshot compare | green |
| 244-01-04 | XBN-01, XBN-02 | scoped quality gate | `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64 scripts/quality_gates.sh` | green |

## Command Results

```bash
node .codex/get-shit-done/bin/gsd-tools.cjs init phase-op 244
```

Result: PASS. Phase 244 context and plan are present.

```bash
node .codex/get-shit-done/bin/gsd-tools.cjs roadmap analyze
```

Result: PASS. Phase 244 is planned with one plan; phases 239-243 are
disk-complete.

```bash
git diff --check
```

Result: PASS.

```bash
scripts/bench.sh --snapshot --compare --suite=kernel_x86_64
```

Initial result before approval: baseline update required. The command configured
and built the suite-scoped benchmark runner, then reported these missing
maintained baselines:

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

No snapshot update was made.

Final result after approved snapshot update and optimized benchmark repair:
PASS.

```bash
EMEL_BENCH_SUITE=kernel_x86_64 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=emel
```

Initial result: PASS. Output was captured in
`/tmp/emel-phase244-kernel-x86-current.sl4mKm.txt`; no snapshot file was
modified.

| Benchmark entry | Candidate ns/op |
|-----------------|-----------------|
| `kernel/x86_64/op_add` | 71.500 |
| `kernel/x86_64/op_cos` | 1450.490 |
| `kernel/x86_64/op_div` | 114.500 |
| `kernel/x86_64/op_dup` | 77.900 |
| `kernel/x86_64/op_log` | 3081.090 |
| `kernel/x86_64/op_mul` | 71.700 |
| `kernel/x86_64/op_mul_mat` | 2584.890 |
| `kernel/x86_64/op_sin` | 1664.700 |
| `kernel/x86_64/op_soft_max` | 4816.490 |
| `kernel/x86_64/op_sqr` | 78.300 |
| `kernel/x86_64/op_sqrt` | 151.600 |
| `kernel/x86_64/op_sub` | 84.600 |
| `kernel/x86_64/op_unary_exp` | 3662.790 |
| `kernel/x86_64/op_unary_neg` | 80.400 |
| `kernel/x86_64/op_unary_relu` | 98.800 |

Source-backed audit then found that the first approved publication only covered
common f32/unary/matmul entries and did not prove the x86_64 optimized flash and
q2/q3/q6 benchmark lanes. `tools/bench/kernel/x86_64_bench.cpp` was repaired to
add four counter-checked optimized entries:

| Benchmark entry | Proof |
|-----------------|-------|
| `kernel/x86_64/op_flash_attn_ext_decode_like` | Aborts unless `optimized_flash_dispatch_count()` increments and `shared_flash_dispatch_count()` does not |
| `kernel/x86_64/op_mul_mat_q2_k_q8_k` | Aborts unless `optimized_q2_dispatch_count()` increments and `shared_q2_dispatch_count()` does not |
| `kernel/x86_64/op_mul_mat_q3_k_q8_k` | Aborts unless `optimized_q3_dispatch_count()` increments and `shared_q3_dispatch_count()` does not |
| `kernel/x86_64/op_mul_mat_q6_k_q8_k` | Aborts unless `optimized_q6_dispatch_count()` increments and `shared_q6_dispatch_count()` does not |

```bash
EMEL_BENCH_SUITE=kernel_x86_64 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=compare
```

Initial result: PASS. Output was captured in
`/tmp/emel-phase244-kernel-x86-compare.ZbJiE5.txt`; no snapshot file was
modified. After repair, the compare output contains 19 EMEL/reference rows,
including optimized flash and q2/q3/q6 entries, proving the reference/shared
lane remains separate from the EMEL-owned optimized benchmark lane.

Representative compare rows from that run:

| Benchmark entry | EMEL ns/op | Reference ns/op | Ratio |
|-----------------|------------|-----------------|-------|
| `kernel/x86_64/op_add` | 125.700 | 397.700 | 0.316x |
| `kernel/x86_64/op_mul_mat` | 2633.790 | 7301.070 | 0.361x |
| `kernel/x86_64/op_soft_max` | 4898.180 | 1132.900 | 4.324x |
| `kernel/x86_64/op_unary_exp` | 3773.590 | 2059.090 | 1.833x |

```bash
EMEL_BENCH_SUITE=kernel_x86_64 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=emel
EMEL_BENCH_SUITE=kernel_x86_64 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=compare
```

Result: PASS. The optimized EMEL smoke emitted 19 `kernel/x86_64/*` entries,
including the four counter-checked optimized entries. The compare smoke emitted
matching EMEL/reference rows for all 19 entries.

```bash
scripts/bench.sh --snapshot --update --suite=kernel_x86_64
```

Result: PASS after explicit user approval. The benchmark snapshot baseline was
merged into `snapshots/bench/benchmarks.txt` and later refreshed after the
source-backed audit repair to include all 19 maintained `kernel/x86_64/*`
entries.

```bash
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=10 --write-generation-baseline snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10.txt
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=100 --write-generation-baseline snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100.txt
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=1000 --write-generation-baseline snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000.txt
```

Result: PASS after explicit user approval. The maintained generation
publication baselines were updated in `snapshots/parity/`.

```bash
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=10 --write-generation-baseline /tmp/emel-phase244-baselines.N7inir/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10.txt
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=100 --write-generation-baseline /tmp/emel-phase244-baselines.N7inir/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100.txt
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=1000 --write-generation-baseline /tmp/emel-phase244-baselines.N7inir/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000.txt
```

Result: PASS. These writes targeted `/tmp`, not checked-in snapshots.

| Max tokens | Generated tokens | Output bytes | Optimized flash | Optimized q6 | Shared q6 |
|------------|------------------|--------------|-----------------|--------------|-----------|
| 10 | 10 | 20 | 228 | 390 | 0 |
| 100 | 100 | 248 | 768 | 1380 | 0 |
| 1000 | 1000 | 2498 | 6168 | 11280 | 0 |

Candidate-vs-snapshot diff summary:

| Max tokens | Stale snapshot | Candidate |
|------------|----------------|-----------|
| 10 | `output_length=22`, `trace_token_count=0` | `output_length=20`, `trace_token_count=10`, token IDs and score gaps populated |
| 100 | `output_length=277`, `trace_token_count=0` | `output_length=248`, `trace_token_count=100`, token IDs and score gaps populated |
| 1000 | `output_length=2866`, `trace_token_count=0` | `output_length=2498`, `trace_token_count=1000`, token IDs and score gaps populated |

## Validation Sign-Off

- [x] Phase 244 context and plan exist.
- [x] Benchmark preflight was run without snapshot updates.
- [x] Missing `kernel/x86_64/*` benchmark baseline entries are identified.
- [x] Candidate benchmark entries and EMEL/reference compare rows were captured
  under `/tmp` without modifying `snapshots/bench/`.
- [x] Stale maintained generation publication baselines are identified from
  Phase 243 live parity evidence.
- [x] Candidate generation publication baselines were written to `/tmp` and
  diffed against checked-in stale snapshots without modifying `snapshots/`.
- [x] Benchmark snapshot update is approved and applied.
- [x] Source-backed audit gap for `XBN-01` is repaired by counter-checked
  optimized flash and q2/q3/q6 benchmark entries.
- [x] Maintained generation baseline updates are approved and applied.
- [x] Scoped quality gate passes after approved publication updates.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set in
  frontmatter.
- [x] Rule-compliance evidence is recorded through suite-scoped benchmark
  commands, EMEL/reference lane separation, publication snapshot diffs, and the
  scoped quality gate.

**Approval:** granted by user; snapshots updated.
