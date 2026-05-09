# Phase 242: Suspension-Safe Request and Progress Ownership - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Define the async I/O request/progress ownership contract before the strategy can suspend or resume.
This phase validates source, target, progress, callback, and scheduler contracts, but still fails
closed before any actual async progress work.

</domain>

<decisions>
## Implementation Decisions

### Stable Storage
- Introduce caller-owned request/progress storage in `emel::io::async::event`.
- Public `load_window` continues to carry required references only.
- Context remains empty; no request, target, callback, or progress fields are mirrored into actor
  context.

### Validation Shape
- Model runtime validation outcomes as guards and explicit states in `sm.hpp`.
- Source contract covers non-zero logical bytes and no offset overflow.
- Target contract covers non-null target buffer and enough target-window bytes.
- Progress contract covers progress not exceeding the logical byte length.
- Scheduler contract is compile-time/static for the selected `emel::co_sm` scheduler policy.

### Deferred Runtime
- Valid requests still end in `unsupported_strategy` until Phase 243 adds partial progress,
  success, and richer terminal errors.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/io/staged_read` already models validation decisions through guards and explicit
  transition states.
- `src/emel/io/async` currently has a fail-closed boundary and no persistent context.

### Integration Points
- `src/emel/io/async/events.hpp`
- `src/emel/io/async/errors.hpp`
- `src/emel/io/async/guards.hpp`
- `src/emel/io/async/actions.hpp`
- `src/emel/io/async/sm.hpp`
- `tests/io/loader/lifecycle_tests.cpp`

</code_context>

<specifics>
## Specific Ideas

Tests should prove invalid source/target/progress paths report deterministic errors, valid requests
still fail closed as unsupported, and no actor context fields retain request/progress/callback data.

</specifics>

<deferred>
## Deferred Ideas

Phase 243 owns actual bounded suspend/resume progress, partial outcomes, terminal success, and the
full async error taxonomy beyond validation and unsupported-boundary behavior.

</deferred>
