---
phase: 21-benchmark-attribution-and-impact
verified: 2026-03-23T09:30:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 21 Verification Report

**Phase Goal:** Publish maintained benchmark evidence that proves and measures the vectorized
quantized path against the current v1.3 scalar baseline.
**Verified:** 2026-03-23T09:30:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The maintained canonical ARM compare workload now publishes quantized attribution that distinguishes optimized q2/q3/q6 execution from shared scalar row helpers. | ✓ VERIFIED | [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp) now captures q2/q3/q6 optimized/shared counts for the canonical generation benchmark case, and [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp) prints `# generation_quantized_evidence:` while rejecting false AArch64 claims. |
| 2 | Maintained compare evidence now republishes `1`, `10`, `100`, and `1000` token generation results against the current v1.3 baseline. | ✓ VERIFIED | [benchmarks_compare.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks_compare.txt), [benchmarks.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks.txt), and [generation_compare_current.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/testdata/generation_compare_current.txt) all now include maintained `1/10/100/1000` generation entries. |
| 3 | At least one maintained generation length now shows measurable end-to-end improvement over the preserved v1.3 baseline without overstating slower cases. | ✓ VERIFIED | [benchmarks.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/docs/benchmarks.md) now publishes `max_tokens=1` at `0.632x` versus llama.cpp and computes a preserved-baseline speedup of `3.560x`, while the same page truthfully reports `10/100/1000` as slower than the current reference run. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tools/bench/generation_bench.cpp` | Canonical q2/q3/q6 benchmark attribution capture | ✓ EXISTS + SUBSTANTIVE | Captures optimized/shared q2/q3/q6 dispatch counts for the canonical generation benchmark case. |
| `tools/bench/bench_main.cpp` | Compare-mode quantized evidence publication and proof | ✓ EXISTS + SUBSTANTIVE | Prints `generation_quantized_evidence` and enforces strict AArch64 optimized/shared expectations. |
| `tools/docsgen/docsgen.cpp` | Generated benchmark docs surface quantized evidence | ✓ EXISTS + SUBSTANTIVE | Parses the new compare metadata and publishes a quantized evidence section. |
| `snapshots/bench/benchmarks.txt` | Refreshed maintained generation snapshot entries | ✓ EXISTS + SUBSTANTIVE | Includes `1/10/100/1000` generation entries under the maintained gate environment. |
| `snapshots/bench/benchmarks_compare.txt` | Refreshed compare evidence and attribution headers | ✓ EXISTS + SUBSTANTIVE | Includes flash and quantized evidence headers plus widened generation rows. |
| `tools/bench/testdata/generation_compare_current.txt` | Refreshed widened maintained generation compare artifact | ✓ EXISTS + SUBSTANTIVE | Publishes all maintained generation compare rows. |
| `docs/benchmarks.md` | Generated docs align with maintained compare snapshot | ✓ EXISTS + SUBSTANTIVE | Publishes both flash and quantized evidence plus widened generation rows. |

**Artifacts:** 7/7 verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| BENCH-08 | ✓ SATISFIED | - |
| BENCH-09 | ✓ SATISFIED | - |

## Residual Risks

1. **Non-generation benchmark variance remains noisy**
   - Issue: the full
     [quality_gates.sh](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/scripts/quality_gates.sh)
     run still emitted benchmark variance warnings on unrelated cases such as
     `batch/planner_equal`, `memory/recurrent_full`, `text/encoders/spm_short`, and
     `tokenizer/preprocessor_plamo2_long`.
   - Impact: repo-wide benchmark noise is still present even after the maintained generation and
     quantized publication surface is refreshed.
   - Recommendation: treat this as broader benchmark-baseline stability debt rather than as a
     failure of Phase 21’s maintained quantized publication goals.

## Automated Checks

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare | sed -n '1,6p'`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 build/bench_tools_ninja/bench_runner --mode=compare > /tmp/bench_compare_phase21.txt`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 build/bench_tools_ninja/bench_runner --mode=emel > /tmp/bench_emel_phase21.txt`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 BENCH_TOLERANCE=0.30 scripts/bench.sh --snapshot --compare`
- `build/docsgen/docsgen --root . --check`
- `scripts/quality_gates.sh` ✓ exits `0`; benchmark step still warns and is explicitly tolerated by the script

## Verification Notes

- The maintained benchmark publication surface now has truthful quantized attribution.
- The widened generation compare evidence is published across snapshots, docs, and maintained
  compare artifacts.
- `max_tokens=1` is materially faster than the preserved v1.3 flash baseline, while longer decode
  lengths are still reported honestly as slower than the current reference run.

---
*Verified: 2026-03-23T09:30:00Z*
*Verifier: the agent*
