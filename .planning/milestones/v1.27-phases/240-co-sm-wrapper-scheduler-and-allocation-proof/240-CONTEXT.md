# Phase 240: co_sm Wrapper, Scheduler, and Allocation Proof - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the opt-in project-owned `emel::co_sm` surface and tests for scheduler contracts,
fixed-capacity allocation, deterministic public dispatch, and no allocation on the immediate
cooperative path. This phase does not introduce an async I/O strategy component or tensor
integration.

</domain>

<decisions>
## Implementation Decisions

### Wrapper Scope
- Add `emel::co_sm` in `src/emel/sm.hpp` parallel to `emel::sm`.
- Preserve synchronous `emel::sm` behavior and aliases.
- Expose EMEL-owned policy aliases for upstream scheduler and allocator policies.

### Allocation
- Use a project-owned fixed coroutine allocator as the EMEL default so pool exhaustion or oversize
  frames do not fall back to heap allocation.
- Prove the immediate FIFO scheduler path does not allocate coroutine frames.

### Testing
- Add focused doctests under the existing `tests/sm/sm_policy_tests.cpp` shard to avoid snapshot
  churn.
- Cover scheduler static contracts, fixed allocator behavior, sync dispatch normalization, inline
  async dispatch, default FIFO immediate dispatch, and context injection.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Upstream `stateforward/sml/utility/co_sm.hpp` is available through the fetched
  `stateforward_sml` dependency.
- Existing `tests/sm/sm_policy_tests.cpp` proves wrapper and SML policy behavior.

### Established Patterns
- `src/emel/sm.hpp` owns the project wrapper surface.
- Doctests use public dispatch and SML `is(...)` state inspection.

### Integration Points
- `src/emel/sm.hpp`
- `tests/sm/sm_policy_tests.cpp`
- `CMakeLists.txt`

</code_context>

<specifics>
## Specific Ideas

The wrapper should normalize synchronous `process_event(...)` results like `emel::sm`; async
normalization can be revisited when real async I/O events define their result contract.

</specifics>

<deferred>
## Deferred Ideas

Async I/O strategy component, owned async progress state, tensor integration, and public runtime
reporting are deferred to later phases.

</deferred>
