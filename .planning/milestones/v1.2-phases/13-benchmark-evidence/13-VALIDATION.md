---
phase: 13
slug: benchmark-evidence
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
---

# Phase 13 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | existing `bench_runner` plus shell/CLI artifact validation and one manual approval checkpoint |
| **Config file** | `tools/bench/CMakeLists.txt`, `scripts/bench.sh`, `tools/docsgen/docsgen.cpp`, `docs/templates/benchmarks.md.j2`, `docs/benchmarking.md` |
| **Quick run command** | `sh -c 'OUT=/tmp/emel_phase13_compare.txt; EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare >"$OUT" 2>&1 && rg "^# reference_impl: source=cmake_fetch ref=" "$OUT" && rg "^# generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 flash_dispatch_calls=[1-9][0-9]* emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0$" "$OUT" && rg "^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio=" "$OUT"'` |
| **Full suite command** | `scripts/quality_gates.sh && scripts/generate_docs.sh --check` |
| **Estimated runtime** | ~600 seconds |

---

## Sampling Rate

- **After every task commit:** Run the task's exact `<automated>` command from the plan file.
- **After wave 1:** Run the compare-surface proof command, the comparator fixture command, and the `docs/benchmarking.md` runbook grep together.
- **After wave 2:** Record the explicit `approve` or `decline` checkpoint result before any snapshot artifact change.
- **Before `$gsd-verify-work`:** `scripts/quality_gates.sh`, `scripts/generate_docs.sh --check`, the preserved-baseline comparator command, and the generated-doc grep must all be green.
- **Max feedback latency:** 600 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | BENCH-01 | reference-proof | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare 2>&1 | rg '^# reference_impl: source=cmake_fetch ref='` | ✅ | ⬜ pending |
| 13-01-02 | 01 | 1 | BENCH-01 | flash-proof | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare 2>&1 | rg '^# generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 flash_dispatch_calls=[1-9][0-9]* emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0$'` | ✅ | ⬜ pending |
| 13-02-01 | 02 | 1 | BENCH-03 | comparator-fixtures | `sh -c 'python3 tools/bench/compare_flash_baseline.py --baseline tools/bench/testdata/generation_pre_flash_baseline_pass.txt --current tools/bench/testdata/generation_compare_current.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 | rg "^case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 baseline_emel_ns=[0-9.]+ current_emel_ns=[0-9.]+ speedup=[0-9.]+x latency_drop_pct=[0-9.]+$" && ! python3 tools/bench/compare_flash_baseline.py --baseline tools/bench/testdata/generation_pre_flash_baseline_fail.txt --current tools/bench/testdata/generation_compare_current.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 >/dev/null 2>&1'` | ❌ W1 | ⬜ pending |
| 13-02-02 | 02 | 1 | BENCH-02 | runbook-contract | `rg 'generation_pre_flash_baseline.txt|generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1|explicit user approval|scripts/bench.sh --compare-update' docs/benchmarking.md` | ✅ | ⬜ pending |
| 13-03-01 | 03 | 2 | BENCH-02 | approval-checkpoint | `manual checkpoint: record explicit user response approve/decline; automated guard: rg 'NEVER update snapshots without explicit user consent' AGENTS.md` | ✅ manual | ⬜ pending |
| 13-04-01 | 04 | 3 | BENCH-02, BENCH-03 | baseline-artifact | `python3 tools/bench/compare_flash_baseline.py --baseline snapshots/bench/generation_pre_flash_baseline.txt --current snapshots/bench/benchmarks_compare.txt --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 | rg '^case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 baseline_emel_ns=[0-9.]+ current_emel_ns=[0-9.]+ speedup=[0-9.]+x latency_drop_pct=[0-9.]+$'` | ❌ W3 | ⬜ pending |
| 13-04-02 | 04 | 3 | BENCH-02 | docs-publication | `scripts/generate_docs.sh --check && rg 'generation_pre_flash_baseline.txt|Current Flash Evidence|Pre-Flash Baseline Comparison|generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1|speedup=[0-9]+\\.[0-9]+x|latency_drop_pct=[0-9]+\\.[0-9]+' docs/benchmarks.md` | ❌ W3 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing plan/task coverage supplies the required comparator fixtures, preserved baseline artifact
contract, compare-surface proof checks, and approval-gate handling. No separate Wave 0 bootstrap
outside Plans 13-01 through 13-04 is required.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Snapshot publication approval gate | BENCH-02 | `AGENTS.md` requires explicit user consent before any checked-in snapshot update | During Plan 13-03, show the user the exact files `snapshots/bench/generation_pre_flash_baseline.txt`, `snapshots/bench/benchmarks_compare.txt`, and `docs/benchmarks.md`, then stop until the user answers `approve` or `decline`. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or an explicit manual-checkpoint dependency
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 600s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
