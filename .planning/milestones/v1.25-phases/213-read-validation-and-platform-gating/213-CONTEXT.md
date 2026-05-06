---
phase: 213-read-validation-and-platform-gating
status: ready
created: 2026-05-05T15:10:00Z
requirements:
  - READ-02
  - PLAT-01
---

# Phase 213: Read Validation and Platform Gating - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning
**Mode:** Autonomous single-pass context

<domain>
## Phase Boundary

Phase 213 must turn the Phase 212 `src/emel/io/read` boundary into an explicit
validation and platform-gating actor. It may add guards, states, error categories, and
tests for request, file path/file index, length, layout, target-buffer, and platform
preconditions. It must not add file open, seek, read, close, transient lifetime,
copied-byte success, tensor integration, public runtime selection, benchmark claims, or
publication closeout behavior.

</domain>

<decisions>
## Implementation Decisions

- Use `src/emel/io/mmap` as the local validation-chain reference, but keep read-specific
  behavior narrower and copy-oriented.
- Treat `event::read_tensor` without `on_done` as invalid for a potentially successful
  read because callers need the copied-byte outcome. `on_error` remains optional.
- Treat empty/too-long/embedded-NUL file paths and invalid target-buffer spans as
  `invalid_request`.
- Treat out-of-range file index, oversized read length, and offset+length overflow as
  `unsupported_resource`.
- Add `EMEL_IO_READ_PLATFORM_SUPPORTED` compile-time platform gating in `errors.hpp`.
  Supported-platform requests may reach `state_read_attempt_decision`, but Phase 213
  still fails closed there with `unsupported_resource` until Phase 214 supplies concrete
  execution.

</decisions>

<code_context>
## Existing Code Insights

- Phase 212 already added `src/emel/io/read/{context,events,errors,guards,actions,detail,sm}.hpp`
  and `tests/io/read/lifecycle_tests.cpp`.
- `src/emel/io/mmap/guards.hpp` and `src/emel/io/mmap/sm.hpp` provide the closest
  validation-chain pattern.
- `read::action::context` remains empty; dispatch-local request/status data stays in
  `detail::read_tensor_runtime` and `detail::read_attempt_status`.

</code_context>

<specifics>
## Specific Ideas

Add explicit destination-first transitions for request span, file path, file index,
length, layout, target buffer, and platform. Tests should continue driving only
`process_event(...)` and inspecting SML state with `is(...)` / `visit_current_states(...)`.

</specifics>

<deferred>
## Deferred Ideas

Phase 214 owns concrete file open/seek/read/close behavior, short-read detection,
copied-byte `_done`, and full ERR-01 taxonomy.

</deferred>
