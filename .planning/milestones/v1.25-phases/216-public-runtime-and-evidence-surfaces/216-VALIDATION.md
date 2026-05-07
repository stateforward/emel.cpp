---
phase: 216-public-runtime-and-evidence-surfaces
status: passed
validated: 2026-05-05T18:36:06Z
nyquist_compliant: true
requirements:
  - TIO-03
  - VAL-04
---

# Phase 216 Validation

## Nyquist Result

Compliant. Phase 216 makes maintained runtime/tool evidence depend on public
model-loader events emitted by the actual EMEL load path, not on actor-internal
inspection or tool-only read scaffolds.

## Evidence

| Check | Result |
|-------|--------|
| Public model-loader evidence | Passed. `load_done` carries `used_io_strategy`; `load_error` carries requested and used strategy evidence. |
| Actual read-backed path | Passed. Model-loader marks strategy usage only after the explicit `io_load_done_all` RTC transition. |
| Maintained tool binding | Passed. Generation, Sortformer diarization, embedded probe, and paritychecker lanes use `bind_model_load_io_strategy(...)` with public `io::loader::sm` injection. |
| Callback reach-through | Passed. The former `process_event(capture)` load-callback probe pattern is absent from all four maintained tool lanes. |
| Tool output | Passed. Generation and diarization benchmark notes and paritychecker formatter contract output report `load_strategy=<name>` from public load evidence. |
| Tool builds/tests | Passed. Generation, diarization, paritychecker, and embedded probe tool targets compile; generation, diarization, and paritychecker compare tests pass. |
| Quality gate | Passed. The changed-file scoped gate ran relevant generation and diarization benchmark suites, full paritychecker, docsgen, SML scan, and changed-file coverage at 92.8% line / 56.5% branch. |
| Scope | Passed. No low-level read logic was added to tools or `model/loader`, and no staged/chunked, async, device, mmap-runtime, or model-family widening behavior was claimed. |

## Notes

Phase 217 owns broader behavior/source guardrails and naming clarity for the read/copy
strategy so v1.25 cannot be mistaken for the deferred staged/chunked constrained-memory
policy.
