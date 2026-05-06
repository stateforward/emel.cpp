---
phase: 217-behavior-tests-and-scope-guardrails
status: complete
type: context
created: 2026-05-05T18:44:12Z
requirements:
  - VAL-01
  - VAL-02
---

# Phase 217 Context

Phase 217 closes the behavior-test and scope-guardrail gaps from the v1.25 audit.
The read/copy runtime path exists across `io/read`, `io/loader`, `model/tensor`,
`model/loader`, and maintained tools. This phase makes that behavior hard to
regress and removes public naming that could imply the deferred staged/chunked
policy.

## Source Truth

- `io/read` owns copy behavior and deterministic read errors.
- `io/loader` owns strategy selection and dispatch into `io/read`.
- `model/tensor` owns tensor residency after read/copy success.
- `model/loader` and tools must use public runtime events, not direct `io/read`
  events or actor internals.

## Non-Goals

- Do not add staged/chunked constrained-memory, async, device, mmap-runtime, or
  model-family behavior.
- Do not broaden benchmark or model fixture scope.
- Do not weaken existing mmap or loader tests while adding read/copy guardrails.
