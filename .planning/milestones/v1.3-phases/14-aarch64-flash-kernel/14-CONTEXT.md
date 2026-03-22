# Phase 14: AArch64 Flash Kernel - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 14 delivers a backend-specific AArch64 `op_flash_attn_ext` implementation for the maintained
canonical Llama-68M flash-attention contract inside `src/emel/kernel`, preserving reusable
workspace semantics and zero-allocation dispatch. This phase does not widen workload scope, does
not change the existing Boost.SML orchestration contract, and does not yet claim shipped runtime,
paritychecker, or benchmark completion.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- All implementation choices are at the agent's discretion; this is a pure infrastructure phase.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/aarch64/actions.hpp` already dispatches `op_flash_attn_ext` and owns
  `flash_attn_workspace` through the backend context.
- `src/emel/kernel/detail.hpp` already contains the maintained shared scalar
  `run_flash_attn_ext_with_workspace(...)` helper and request-shape validation logic.
- `tests/kernel/aarch64_tests.cpp`, `tests/kernel/lifecycle_tests.cpp`, and
  `tests/kernel/test_helpers.hpp` already cover canonical flash-attention execution and workspace
  reuse behavior.

### Established Patterns
- Backend machines keep the existing `src/emel/kernel/*/sm.hpp` dispatch contract unchanged while
  data-plane specialization lives in backend-local helpers and actions.
- Unsupported requests are rejected explicitly instead of silently claiming optimized execution.
- Persistent workspace ownership stays in backend context and must remain allocation-free during
  dispatch.

### Integration Points
- `src/emel/kernel/aarch64/actions.hpp` is the current seam where AArch64 flash requests still
  route through the shared scalar workspace helper.
- `src/emel/kernel/aarch64/context.hpp` and `src/emel/kernel/detail.hpp` define the current
  reusable workspace contract that the optimized path must preserve.
- `tests/kernel/aarch64_tests.cpp` and `tests/kernel/lifecycle_tests.cpp` are the kernel-facing
  proof surfaces for correctness and workspace reuse in this phase.

</code_context>

<specifics>
## Specific Ideas

No specific requirements; infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

---
*Phase: 14-aarch64-flash-kernel*
*Context gathered: 2026-03-22*
