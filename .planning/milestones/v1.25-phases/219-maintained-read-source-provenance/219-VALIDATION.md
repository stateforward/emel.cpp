---
phase: 219-maintained-read-source-provenance
status: passed
validated: 2026-05-05T21:15:00Z
nyquist_compliant: true
requirements:
  - PLAT-01
  - TIO-03
  - VAL-04
---

# Phase 219 Validation

## Nyquist Result

Compliant. Phase 219 closes the source-provenance gap for maintained read/copy
evidence by moving full-file source-byte loading out of tool-local helpers and
into the maintained EMEL source contract. Phase 222 later moved that contract
to public `src/emel/io/source/any.hpp`.

## Evidence

| Check | Result |
|-------|--------|
| Maintained source contract | Passed. Phase 222 supersedes the original helper placement: maintained tool lanes now call public `emel::io::source::load_file_bytes` from `src/emel/io/source/any.hpp`. |
| Tool-local read scaffolds | Passed. Source scan over generation, Sortformer, embedded probe, and paritychecker maintained lanes found no `read_file_bytes` helper or call sites. |
| Public runtime evidence | Passed. Maintained lanes continue to bind/report read/copy through public `model/loader` and `io/loader` strategy surfaces. |
| Focused tests | Passed. `emel_tests_model_and_batch`, the diarization request shard, and focused Sortformer modules/output tests passed. |
| Domain boundaries | Passed. `scripts/check_domain_boundaries.sh` exited 0. |
| Scoped quality gate | Passed. `scripts/quality_gates.sh` exited 0 with `EMEL_QUALITY_GATES_CHANGED_FILES` scoped to Phase 219 source, test, tool, and planning files. |
| Diarization hang investigation | Passed for Phase 219. The long-running Sortformer parity test had already passed all source-loading fixture assertions and was sampled in runtime encoder/kernel compute, not in Phase 219 source-byte loading. |

## Residual Risk

The full diarization shard remains expensive under the debug/UBSan test binary
and rapid repeated launches can still hit an intermittent macOS dyld cache
startup failure. Those issues are tracked here as validation environment/runtime
concerns, not as Phase 219 source-provenance regressions.
