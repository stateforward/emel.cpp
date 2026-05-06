---
phase: 214-read-execution-errors-and-lifetime
status: ready
created: 2026-05-05T15:35:00Z
requirements:
  - READ-03
  - LIFE-01
  - ERR-01
---

# Phase 214: Read Execution, Errors, and Lifetime - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning
**Mode:** Autonomous single-pass context

<domain>
## Phase Boundary

Phase 214 owns concrete read/copy execution below the Phase 213 validation and platform
gates. It may add component-local platform open/seek/read/close implementation,
same-RTC status fields, read success publication, and deterministic read execution error
outcomes. It must not integrate with `model/tensor`, modify loader selection, publish
benchmark/parity claims, or add staged/async/device strategy behavior.

</domain>

<decisions>
## Implementation Decisions

- Implement platform-local execution in `src/emel/io/read/actions.cpp`, mirroring the
  existing `io/mmap/actions.cpp` convention for platform C APIs.
- Keep dispatch-local file handle, bytes copied, and open/seek/read booleans in
  `detail::read_attempt_status`, not `read::action::context`.
- Close the transient OS resource in the read action before any `_done` publication is
  structurally reachable.
- Surface success through `events::read_tensor_done` with copied byte count and the
  caller-owned target pointer; surface failures through existing `_error` callback.

</decisions>

<code_context>
## Existing Code Insights

- Phase 213 already validates request span, file path, file index, length, layout,
  target buffer, and platform support before `state_read_attempt_decision`.
- `src/emel/io/mmap/actions.cpp` is the closest local platform helper pattern.
- `tests/io/read/lifecycle_tests.cpp` already drives the actor through public
  `process_event(...)` and can add file-backed success/failure cases.

</code_context>

<specifics>
## Specific Ideas

Use open -> seek -> read/close states. Route open failure, seek failure, read failure,
and short read to explicit error states. On success, publish `_done` only after the
resource has been closed.

</specifics>

<deferred>
## Deferred Ideas

Phase 215 owns tensor-side consumption and residency semantics.

</deferred>
