---
phase: 179-v1-20-closeout-evidence-reproducibility-repair
plan: 01
completed: 2026-05-02
status: complete
requirements-addressed:
  - VAL-01
  - VAL-03
---

# Phase 179 Plan 01 Summary

Repaired the v1.20 closeout evidence path so the bench-tool validation is reproducible from a
maintained unfiltered build command and suite-filtered benchmark runs cannot leave the canonical
bench-tools cache in a filtered state.

Implementation changes:

- `scripts/bench.sh` now has `--test-tools`, which configures `build/bench_tools_ninja`
  without a suite filter, builds `bench_runner_tests` and `quality_gates_tests`, and runs the
  focused ctest regex used by closeout validation.
- Suite-filtered benchmark runs now default to suite-specific build directories such as
  `build/bench_tools_ninja_generation`, preserving the canonical bench-tools build for unfiltered
  validation.
- Scoped snapshot updates merge the selected suite into the full benchmark baseline instead of
  replacing unrelated suites, and scoped compare-baseline updates are rejected.
- `tools/bench/reference_ref.txt` is pinned to
  `c5a3bc39b1b0fe56954c6adb99e89b25d5e7b9cb` so closeout benchmark snapshots are not tied to a
  moving upstream branch.
- Phase 172 and Phase 178 closeout artifacts are marked superseded so they cannot be used as the
  authoritative VAL-03 completion claim.

Validation passed:

- `bash -n scripts/bench.sh`
- `scripts/bench.sh --test-tools`
- `ctest --test-dir build/bench_tools_ninja -R 'quality_gates_tests|bench_runner_tests' --output-on-failure`
- changed-file scoped `scripts/quality_gates.sh`
- `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh`

The final full quality gate completed in 1110 seconds with coverage at 91.6% lines and 56.9%
branches, and kept benchmark, paritychecker, fuzz smoke, lint snapshot, and docs lanes enabled.
