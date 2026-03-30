---
phase: 25-quantized-attribution-and-impact
verified: 2026-03-25T22:00:45Z
status: passed
score: 3/3 phase truths verified
---

# Phase 25 Verification Report

**Phase Goal:** Publish maintained benchmark attribution that shows the end-to-end impact of full
quantized-path closure and honestly isolates the next bottleneck.
**Verified:** 2026-03-25T22:00:45Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Maintained benchmark compare output now publishes the shipped runtime contract and fails if the canonical benchmark case drifts away from the approved `8/4/0/0` contract. | ✓ VERIFIED | [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp) now records runtime-contract counts from the shipped generator wrapper, and [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp) prints `generation_runtime_contract:` while hard-failing any canonical contract mismatch. |
| 2 | Stored compare evidence and generated benchmark docs now publish the approved runtime contract and keep the dense-f32-by-contract seams explicit instead of overstating the supported path. | ✓ VERIFIED | [benchmarks_compare.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/bench/benchmarks_compare.txt) now stores `generation_runtime_contract: ... native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`, and [benchmarks.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/docs/benchmarks.md) mirrors that line plus a contract-summary sentence describing the approved dense-f32 seams honestly. |
| 3 | Phase 25 finished through the approval-gated publication workflow and the full repo gate stayed green under the current warning-only benchmark policy. | ✓ VERIFIED | Stored benchmark artifacts were refreshed only after explicit user approval in-session, and [timing.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/snapshots/quality_gates/timing.txt) was restored after a final `scripts/quality_gates.sh` pass exited `0` with warning-only benchmark regressions. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tools/bench/generation_bench.cpp` | Benchmark-side capture of shipped runtime-contract counts | ✓ EXISTS + SUBSTANTIVE | Canonical generation capture now stores the runtime contract alongside existing flash and quantized evidence. |
| `tools/bench/bench_main.cpp` | Compare-mode publication and validation of the approved runtime contract | ✓ EXISTS + SUBSTANTIVE | Compare output now prints `generation_runtime_contract:` and rejects canonical contract drift. |
| `tools/docsgen/docsgen.cpp` | Docs-generation support for runtime-contract metadata and contract summary | ✓ EXISTS + SUBSTANTIVE | Generated docs now require and publish the runtime-contract line plus honest contract-summary language. |
| `snapshots/bench/benchmarks_compare.txt` | Refreshed stored compare evidence | ✓ EXISTS + REFRESHED | The stored compare artifact now includes the canonical `generation_runtime_contract:` metadata. |
| `docs/benchmarks.md` | Regenerated benchmark publication with truthful post-closure attribution | ✓ EXISTS + REFRESHED | The docs now surface the stored runtime-contract line and the approved dense-f32-by-contract explanation. |
| `snapshots/quality_gates/timing.txt` | No leftover generated timing churn after full-gate verification | ✓ RESTORED | The full gate was run for Phase 25, then the generated timing snapshot was restored to the preserved baseline values. |

**Artifacts:** 6/6 verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| BENCH-10 | ✓ SATISFIED | - |

## Gaps Summary

No remaining phase-local gaps.

All v1.5 phases are now complete. The next workflow step is milestone audit and archival.

Current non-blocking benchmark warnings observed during the final full gate were:
- `logits/sampler_raw/vocab_32000`
- `kernel/aarch64/op_soft_max`

Those regressions remain warning-only under the existing `quality_gates.sh` policy and do not
block truthful Phase 25 publication.

## Automated Checks

- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 scripts/bench.sh --compare`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 scripts/bench.sh --compare-update`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh` ✓ exits `0`

## Verification Notes

- Phase 25 preserved the approved dense-f32-by-contract seams as visible contract stages rather
  than collapsing into a misleading "fully quantized everywhere" claim.
- Snapshot and docs publication ran only after explicit user approval, matching the repo rule
  against unapproved snapshot churn.
- Phase 25 closed without changing any SML transition table, actor ownership, or public C API
  boundary.

---
*Verified: 2026-03-25T22:00:45Z*
*Verifier: the agent*
