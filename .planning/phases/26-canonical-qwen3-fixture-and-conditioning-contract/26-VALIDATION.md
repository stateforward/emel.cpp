---
phase: 26
slug: canonical-qwen3-fixture-and-conditioning-contract
status: validated_partial
nyquist_compliant: false
wave_0_complete: true
created: 2026-03-28
---

# Phase 26 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest via CMake/CTest |
| **Config file** | `CMakeLists.txt` + `tools/paritychecker/CMakeLists.txt` |
| **Quick run command** | `./build/coverage/emel_tests_bin --test-case='*formatter*contract*' --no-breaks` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~25 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./build/coverage/emel_tests_bin --test-case='*formatter*contract*' --no-breaks`
- **After every plan wave:** Run `scripts/build_with_zig.sh && ctest --test-dir build/zig --output-on-failure -R emel_tests && ctest --test-dir build/zig --output-on-failure -R lint_snapshot`
- **After maintained tool-surface tasks:** Run `sh -lc 'set +e; EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only > /tmp/phase26_bench.txt 2>&1; status=$?; cat /tmp/phase26_bench.txt; test $status -ne 0; rg "Qwen3-0\\.6B-Q8_0\\.gguf|prepare_emel_fixture|generation_formatter_contract" /tmp/phase26_bench.txt'`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 25 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 26-01-01 | 01 | 1 | FIX-01 | subprocess / unit | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ | ✅ green |
| 26-01-02 | 01 | 1 | FIX-01 | maintained bench pre-runtime failure | `sh -lc 'set +e; EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only > /tmp/phase26_fixture_bench.txt 2>&1; status=$?; cat /tmp/phase26_fixture_bench.txt; test $status -ne 0; rg "Qwen3-0\\.6B-Q8_0\\.gguf|prepare_emel_fixture" /tmp/phase26_fixture_bench.txt'` | ✅ | ✅ green |
| 26-02-01 | 02 | 2 | COND-01 | unit | `./build/coverage/emel_tests_bin --test-case='*formatter*contract*,*conditioner*structured*' --no-breaks` | ✅ | ✅ green |
| 26-03-01 | 03 | 3 | COND-01 | unit | `./build/coverage/emel_tests_bin --test-case='*generator*structured*,*generator*message*' --no-breaks` | ✅ | ✅ green |
| 26-04-01 | 04 | 4 | COND-01 | subprocess | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ | ✅ green |
| 26-04-02 | 04 | 4 | COND-01 | maintained bench contract failure path | `sh -lc 'set +e; EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only > /tmp/phase26_formatter_bench.txt 2>&1; status=$?; cat /tmp/phase26_formatter_bench.txt; test $status -ne 0; rg "generation_formatter_contract|Qwen3-0\\.6B-Q8_0\\.gguf" /tmp/phase26_formatter_bench.txt'` | ✅ | ✅ green |
| 26-04-03 | 04 | 4 | COND-01 | repo gate | `scripts/quality_gates.sh` | ✅ | ❌ red |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Resolved formatter-contract string remains operator-readable and semantically correct in maintained parity/bench setup or failure output | COND-01 | Automated tests can assert presence and exact strings, but human review is still needed to confirm the published contract matches the intended maintained semantics | Run maintained parity/bench flows on the canonical fixture, inspect published formatter-contract strings, and confirm they encode the approved primary-template-only contract with `tools=none`, `add_generation_prompt=true`, and `enable_thinking=false` even before runtime bring-up succeeds |
| Full repo gate still reaches successful-generation paritychecker assertions that are outside the approved Phase 26 pre-runtime failure boundary | COND-01 | `scripts/quality_gates.sh` currently fails in `paritychecker_tests` on maintained generation success-path expectations (`generation parity ok`, quantized dispatch metrics, attribution buckets) that Phase 26 intentionally does not satisfy before Phase 27 runtime bring-up | After Phase 27 restores truthful maintained Qwen generation, rerun `scripts/quality_gates.sh` and require the paritychecker suite to pass without reverting the Phase 26 canonical fixture or formatter-contract publication behavior |

---

## Validation Audit 2026-03-27

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 0 |
| Escalated | 1 |

Notes:
- Reconstructed coverage from `26-01` through `26-04` plan and summary artifacts, then reran the phase-level repo gate on March 27, 2026.
- `scripts/quality_gates.sh` failed in the `paritychecker` step because full-suite `paritychecker_tests` still expect successful maintained generation output and runtime attribution that remain downstream of the approved Phase 26 boundary.
- Phase 26 therefore has complete task-level automated verification for its explicit pre-runtime contract, but it is not yet Nyquist-compliant at the repo-gate level.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [ ] Feedback latency < 25s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** validated partial; rerun after Phase 27 restores maintained generation success paths
