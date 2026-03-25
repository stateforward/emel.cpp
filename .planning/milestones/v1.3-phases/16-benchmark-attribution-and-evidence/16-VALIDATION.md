---
phase: 16
slug: benchmark-attribution-and-evidence
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
---

# Phase 16 - Validation Strategy

> Per-phase validation contract for live benchmark attribution and approval-gated publication.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | benchmark compare runner + docsgen + repo gates |
| **Config file** | `tools/bench/generation_bench.cpp`, `tools/bench/bench_main.cpp`, `tools/docsgen/docsgen.cpp` |
| **Quick run command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` |
| **Focused run command** | `python3 tools/bench/compare_flash_baseline.py --baseline ... --current snapshots/bench/benchmarks_compare.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~1200 seconds |

---

## Sampling Rate

- **After live attribution changes:** run the one-iteration compare command
- **After artifact publication changes:** run `scripts/generate_docs.sh --check` plus the baseline
  comparator
- **Before phase closeout:** run `scripts/quality_gates.sh`
- **Max feedback latency:** 1200 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | BENCH-04, BENCH-05 | live compare attribution | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` | ✅ | ⬜ pending |
| 16-01-02 | 01 | 1 | BENCH-04, BENCH-05 | repo regression | `scripts/quality_gates.sh` | ✅ | ⬜ pending |
| 16-02-01 | 02 | 2 | BENCH-04, BENCH-05, BENCH-06 | approval checkpoint | `rg 'NEVER update snapshots without explicit user consent' AGENTS.md` | ✅ | ⬜ pending |
| 16-02-02 | 02 | 2 | BENCH-05, BENCH-06 | snapshot/docs publication | `scripts/generate_docs.sh --check && python3 tools/bench/compare_flash_baseline.py --baseline snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt --current snapshots/bench/benchmarks_compare.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing benchmark, compare, and docsgen infrastructure already exists. No extra scaffold is
required before execution.

---

## Manual-Only Verifications

- Explicit user approval is required before checked-in benchmark snapshots or generated benchmark
  docs change.

---

## Validation Sign-Off

- [x] All tasks have an `<automated>` verification command
- [x] Sampling continuity is preserved across both plans
- [x] The approval gate is explicit
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** pending
