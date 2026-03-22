---
phase: 09
slug: benchmark-integration-hardening
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-10
---

# Phase 09 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | existing `bench_runner` + snapshot/docs shell validation |
| **Config file** | `tools/bench/CMakeLists.txt`, `scripts/bench.sh`, `tools/docsgen/docsgen.cpp` |
| **Quick run command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio='` |
| **Full suite command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --snapshot --compare` |
| **Estimated runtime** | ~420 seconds |

---

## Sampling Rate

- **After every task commit:** Run `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio='`
- **After every plan wave:** Run `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --snapshot --compare`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 420 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | VBEN-01 | compare-smoke | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio='` | ✅ | ⬜ pending |
| 09-01-02 | 01 | 1 | VBEN-01 | docs-proof | `rg 'generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1|scripts/bench.sh --compare|EMEL_BENCH_GENERATION_' docs/benchmarking.md` | ✅ | ⬜ pending |
| 09-02-01 | 02 | 2 | VBEN-02 | snapshot-gate | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --snapshot --compare` | ✅ | ⬜ pending |
| 09-02-02 | 02 | 2 | VBEN-02 | baseline-refresh | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare-update && scripts/generate_docs.sh && rg 'generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1' snapshots/bench/benchmarks_compare.txt docs/benchmarks.md` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Snapshot baseline refresh approval | VBEN-02 | `AGENTS.md` requires explicit user consent before snapshot updates | Stop before any `--update` or `--compare-update` snapshot refresh, ask the user for approval, then continue the plan only if approval is granted. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 420s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
