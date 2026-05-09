# Phase 249: Runtime Scheduler Error Contract Repair - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Make scheduler and resource terminal error semantics reachable through explicit
Stateforward.SML guards and transitions instead of constant scheduler predicates.
This phase closes `AIO-06` and `INT-02` by repairing the runtime error contract in
`src/emel/io/async` and its public async dispatch tests.

</domain>

<decisions>
## Implementation Decisions

### Contract Repair
- Keep scheduler/resource failure selection in `src/emel/io/async/sm.hpp` transition rows.
- Put runtime predicates in `src/emel/io/async/guards.hpp`; do not hide scheduler/resource
  routing in actions or detail helpers.
- Publish deterministic terminal `_error` outcomes for scheduler/resource failure,
  cancellation/rejection, and existing validation failure categories.
- Drive tests through public `process_event(...)` / async dispatch surfaces and SML state
  inspection.

### the agent's Discretion
All implementation details are at the agent's discretion within the SML and allocation
rules. Prefer the smallest repair that makes scheduler/resource failure observable without
weakening existing success and validation behavior.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/io/async/sm.hpp` already has explicit validation, cancellation, scheduler
  decision, progress-kind, terminal done, and terminal error callback states.
- `src/emel/io/async/guards.hpp` currently contains the scheduler decision predicates that
  need runtime meaning.
- `src/emel/io/async/actions.hpp` already publishes `events::load_window_error` through
  callbacks and records synchronous status for public dispatch.
- `tests/io/loader/lifecycle_tests.cpp` contains public async callback helpers and existing
  lifecycle assertions.

### Established Patterns
- Async orchestration uses destination-first transition rows, guard-owned behavior choice,
  and action-only execution of already-selected paths.
- Public async behavior is validated through `emel::io::async::sm`, `process_event(...)`,
  callbacks, and SML `is(...)` state inspection.
- Error codes are exposed through component-owned `error` enums cast to `emel::error::type`.

### Integration Points
- Scheduler/resource failure must connect from `event::load_window` request contracts to
  `detail::load_window_runtime`, then through explicit guards and error states.
- Loader and tensor integration phases depend on async errors remaining public and
  deterministic, not private helper return values.

</code_context>

<specifics>
## Specific Ideas

No specific user-facing requirements. Use ROADMAP success criteria and the existing async
machine structure as the implementation guide.

</specifics>

<deferred>
## Deferred Ideas

Broader device scheduler, platform I/O completion, accelerator, and decode overlap work
remain out of scope for v1.27.

</deferred>
