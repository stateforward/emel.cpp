---
phase: 178-v1-20-closeout-gate-and-evidence-repair
verified: 2026-05-02T00:00:00Z
status: superseded
score: 4/5 phase truths verified
superseded_by: 179-v1-20-closeout-evidence-reproducibility-repair
---

# Phase 178 Verification Report

**Phase Goal:** Resolve the remaining VAL-03 closeout blocker and produce source-backed final
closeout evidence.  
**Status:** superseded

## Goal Achievement

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Full benchmark scope no longer routes through the monolithic timeout-prone runner by default. | passed | `scripts/quality_gates.sh` expands full benchmark scope from `tools/bench/dependency_manifest.txt` into suite-level runs. |
| 2 | Host-specific and internal-only benchmark suites are filtered explicitly. | passed | `bench_suite_supported_for_host()` skips unsupported `kernel_x86_64`, `kernel_aarch64`, and internal `sm_any` cases. |
| 3 | Reference-dependent memory benchmark suites build under filtered suite targets. | passed | `tools/bench/CMakeLists.txt` includes `memory_kv`, `memory_recurrent`, and `memory_hybrid` in the reference-needed suite filter. |
| 4 | Benchmark snapshot changes were explicit and approved. | passed | User explicitly approved updating the benchmark snapshots; `snapshots/bench/benchmarks.txt` records the new baselines. |
| 5 | The final closeout gate passes without weakening required lanes. | superseded | Phase 178 produced a passing full gate, but the later milestone audit found its bench tooling validation command was not reproducible from the current maintained build state. Phase 179 owns the final reproducible closeout proof. |

## Automated Checks

- `bash -n scripts/quality_gates.sh`
- Bench tooling test build and `ctest -R 'quality_gates_tests|bench_runner_tests'`
- `BENCH_TOLERANCE=0.30 scripts/bench.sh --snapshot --compare --suite=memory_kv`
- `BENCH_TOLERANCE=0.30 scripts/bench.sh --snapshot --compare --suite=kernel_aarch64`
- `BENCH_TOLERANCE=0.30 scripts/bench.sh --snapshot --compare --suite=encoder_spm`
- `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh`

## Notes

Phase 177 and Phase 178 are superseded closeout attempts. Phase 179 is the authoritative VAL-01 and
VAL-03 completion evidence.
