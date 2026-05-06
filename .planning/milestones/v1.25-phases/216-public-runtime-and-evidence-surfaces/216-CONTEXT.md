---
phase: 216-public-runtime-and-evidence-surfaces
status: complete
type: context
created: 2026-05-05T18:27:09Z
requirements:
  - TIO-03
  - VAL-04
---

# Phase 216 Context

Phase 216 closes the public runtime and maintained evidence-surface gap from the
v1.25 milestone audit. Phase 215 made tensor-owned read loading available through
public actors; this phase makes maintained runtime and tool entrypoints select and
report that path without actor-internal reach-through.

## Source Truth

- `src/emel/model/loader` is the maintained runtime orchestration surface above
  `model/tensor` and `io/loader`.
- Maintained generation, diarization, embedded probe, and paritychecker tool lanes
  must bind loading strategy through public model-loader request fields.
- Model-loader done/error events are the public place to report whether read/copy,
  mmap, unsupported, or non-I/O loading was actually used.

## Non-Goals

- Do not add low-level read logic to tools or `model/loader`.
- Do not claim staged/chunked constrained-memory, async, device, or new model-family
  behavior.
- Do not infer read-backed evidence from planning artifacts, tensor actor internals,
  or callback-time probe dispatches.
