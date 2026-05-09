# Phase 241: Async I/O Strategy Component Boundary - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Create the dedicated cooperative async I/O strategy component under `src/emel/io/async` and prove
it is separate from shipped synchronous strategies. This phase establishes a fail-closed boundary
only; validation, owned progress state, suspend/resume behavior, and tensor integration land later.

</domain>

<decisions>
## Implementation Decisions

### Component Shape
- Use canonical files: `context`, `errors`, `events`, `guards`, `actions`, `detail`, and `sm`.
- Expose `emel::io::async::sm` and top-level `emel::IoAsync`.
- Use `emel::co_sm` for the component machine.

### Runtime Behavior
- Accept a minimal `event::load_window` request shape but reject it as unsupported until the
  progress contract exists.
- Publish an explicit `_error` event when an error callback is present.
- Return to `state_ready` after every rejected dispatch.

### Test Placement
- Add tests to existing `tests/io/loader/lifecycle_tests.cpp` to avoid snapshot churn while still
  covering the public I/O strategy boundary.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/io/read` and `src/emel/io/staged_read` provide canonical component structure.
- `src/emel/machines.hpp` owns top-level PascalCase aliases.

### Established Patterns
- Strategy events use required request references and same-RTC callbacks.
- Strategy wrappers translate public requests into internal runtime events and status structs.

### Integration Points
- `src/emel/io/async/**`
- `src/emel/machines.hpp`
- `tests/io/loader/lifecycle_tests.cpp`

</code_context>

<specifics>
## Specific Ideas

The component should not modify `io/mmap`, `io/read`, or `io/staged_read`; tests should prove the
new boundary exists and fail-closed semantics are explicit.

</specifics>

<deferred>
## Deferred Ideas

Validation and owned suspension-safe progress state are deferred to Phase 242. Actual partial
progress, resume, success, and error taxonomy beyond unsupported strategy are deferred to Phase 243.

</deferred>
