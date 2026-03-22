---
phase: 07
slug: generation-benchmark-harness
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-08
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | existing `bench_runner` + shell validation |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --emel-only` |
| **Full suite command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` |
| **Estimated runtime** | ~300 seconds |

---

## Sampling Rate

- **After every task commit:** Run `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --emel-only`
- **After every plan wave:** Run `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 300 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | BENCH-01 | build | `cmake --build build/bench_tools_ninja --parallel --target bench_runner` | ✅ | ⬜ pending |
| 07-01-02 | 01 | 1 | BENCH-01, BENCH-02 | smoke | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --emel-only` | ✅ | ⬜ pending |
| 07-02-01 | 02 | 2 | BENCH-02, BENCH-03 | behavior | `build/bench_tools_ninja/bench_runner --mode=emel | rg "generation/"` | ✅ | ⬜ pending |
| 07-02-02 | 02 | 2 | BENCH-01, BENCH-02, BENCH-03 | compare-smoke | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 300s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
