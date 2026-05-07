# Phase 225: Read Closeout Runtime Validation And SML Repair - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the refreshed v1.25 source-backed audit gaps by restoring executable
model/batch validation, repairing maintained read/copy orchestration so it no longer
relies on action-loop `io_loader->process_event(...)` dispatch, and reconciling
closeout artifact paths.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion because this is an
infrastructure and rule-compliance repair phase. Preserve v1.25 scope: no staged/chunked,
async, device, model-family widening, mmap behavior changes, tool-only read scaffolds, or
benchmark-regression overrides.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/model/loader/sm.hpp`, `actions.hpp`, `guards.hpp`, and `events.hpp` own
  model-loader orchestration and public used/requested I/O strategy evidence.
- `src/emel/io/loader/sm.hpp` already routes `strategy_kind::read_copy` to the injected
  `io/read` actor through explicit guards.
- `src/emel/io/read` owns the actor-local read/copy validation and source-span copy.
- `tests/model/loader/lifecycle_tests.cpp`, `tests/model/tensor/lifecycle_tests.cpp`,
  `tests/io/loader/lifecycle_tests.cpp`, and `tests/io/read/lifecycle_tests.cpp` provide
  focused public-dispatch coverage.

### Established Patterns
- SML transition tables use destination-first rows, explicit phase labels, and
  completion transitions for bounded internal phases.
- Runtime behavior choice belongs in `guards.hpp` and `sm.hpp`; actions execute an
  already selected behavior without choosing the next route.
- Maintained read/copy evidence must trace through public runtime paths and avoid
  actor-detail reach-through from tools.

### Integration Points
- `.planning/v1.25-MILESTONE-AUDIT.md` is the source of the refreshed gaps.
- `effect_dispatch_io_loads` in `src/emel/model/loader/actions.hpp` is the current
  SML readiness risk because it loops over tensor effects and dispatches `io_loader`.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  currently aborts in dyld before doctests execute in this environment.

</code_context>

<specifics>
## Specific Ideas

- Prefer explicit model-loader states/transitions or typed completion carriers for
  per-tensor read/copy I/O progress instead of hiding the orchestration loop in an action.
- Keep bulk numeric or byte-copy work in the already-owned kernel/action paths; do not
  add dynamic allocation during SML dispatch.
- Record any persistent macOS dyld launch limitation with source-backed substitute
  evidence only if direct test execution cannot be made healthy.

</specifics>

<deferred>
## Deferred Ideas

Staged/chunked constrained-memory read policy, cooperative async loading,
device-specific loading strategies, model-family widening, and performance optimization
todos remain out of scope.

</deferred>
