---
phase: 225
slug: read-closeout-runtime-validation-and-sml-repair
status: passed_with_dyld_fallback
nyquist_compliant: true
dyld_launch_blocker: true
wave_0_complete: true
created: 2026-05-06
validated: 2026-05-06T15:45:45Z
requirements:
  - VAL-01
  - TIO-03
  - VAL-04
  - VAL-03
---

# Phase 225 — Validation Status

Phase 225 is Nyquist compliant. The direct `build/zig` focused CTest lane still exposes
the known macOS dyld launch blocker before doctests execute, so validation uses the
explicit fallback allowed by Plan 06: exact dyld output is recorded in
`225-VERIFICATION.md`, the same focused doctest shards pass in the coverage build, and
source scans prove the maintained read/copy lane no longer contains the audited hidden
model-loader action loop or actor-internal read reach-through.

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists.
- [x] Phase `VERIFICATION.md` exists.
- [x] Plan 06 command evidence was produced from current source.
- [x] `nyquist_compliant: true` is backed by command evidence, not frontmatter alone.

## Requirement Status

| Requirement | Status | Evidence |
|-------------|--------|----------|
| VAL-01 | compliant with dyld fallback | Direct `build/zig` launch blocker recorded; coverage-built `emel_tests_model_and_batch` and `emel_tests_io` passed through CTest. |
| TIO-03 | compliant | Source scans found no `effect_dispatch_io_loads`, no per-tensor `io_loader->process_event` loop, no maintained `emel/io/read/detail.hpp` or `emel/io/read/events.hpp` reach-through, and no direct `read_tensor_request`. |
| VAL-04 | compliant | Maintained generation, Sortformer, embedded probe, and paritychecker callers all contain request-owned `io_load_spans`, public source loading, and `used_io_strategy` evidence propagation. |
| VAL-03 | compliant | `225-VERIFICATION.md`, this validation file, Plan 06 summary, active audit, and archived audit were refreshed from Plan 06 commands. |

## Command Evidence

| Command | Result |
|---------|--------|
| `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` | dyld launch abort before doctests |
| `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` | passed |
| `ctest --test-dir build/zig --output-on-failure -R 'emel_tests_(model_and_batch\|io)'` | dyld launch abort before doctests |
| `scripts/check_domain_boundaries.sh` | passed |
| `node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` | passed with 16 pre-existing warnings and no errors |
| `EMEL_QUALITY_GATES_CHANGED_FILES='<Phase 225 source/test/tool files>' EMEL_QUALITY_GATES_BENCH_SUITE='generation:diarization_sortformer' scripts/quality_gates.sh` | passed without benchmark-regression override |

The scoped quality gate ran relevant generation and Sortformer benchmark suites, coverage,
paritychecker, docs, and fuzz-smoke selection. Its coverage CTest lane passed both focused
shards and reported 94.8% line coverage and 66.5% branch coverage for scoped changed source
files.

## Source-Backed Fallback

The fallback is valid because the failing direct CTest runs abort before doctest execution
with dyld shared-cache / `libSystem.B.dylib` output. Source scans provide the maintained
path evidence required by Plan 06:

- no `effect_dispatch_io_loads` symbol remains in `src/emel/model/loader`
- no per-tensor `io_loader->process_event(...)` loop remains in `src/emel/model/loader`
- maintained caller files provide `io_load_spans`
- maintained caller files use `emel::io::source::load_file_bytes`
- maintained caller files propagate `.used_io_strategy = ev.used_io_strategy`
- maintained model-loader/tool lanes do not include `emel/io/read/detail.hpp`
- maintained model-loader/tool lanes do not include `emel/io/read/events.hpp`
- maintained model-loader/tool lanes do not construct `read_tensor_request`

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | yes | Runtime choice remains in guards/transitions; no model-loader action-loop orchestration remains in the maintained read/copy lane. |
| `docs/rules/sml.rules.md` | yes | The repair uses batch events and bounded data-plane work, not per-tensor anonymous/completion loops. |
| `docs/rules/cpp.rules.md` | yes | Validation used existing build/test gates and avoided snapshot baseline changes. |

## Validation Sign-Off

- [x] Completion preconditions satisfied.
- [x] Rule-compliance review recorded.
- [x] All Plan 06 automated verification commands ran or recorded explicit dyld fallback.
- [x] Domain boundary, consistency, and relevant changed-file quality gate evidence exists.
- [x] No snapshot baseline command was run.
- [x] No benchmark-regression override was used.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** source-backed Plan 06 validation passed with recorded dyld launch blocker.
