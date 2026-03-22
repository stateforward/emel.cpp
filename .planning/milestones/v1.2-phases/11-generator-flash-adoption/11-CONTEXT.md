# Phase 11: Generator Flash Adoption - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 11 adopts the Phase 10 flash-attention kernel inside the shipped canonical generation path
under `src/emel/generator` without changing Boost.SML orchestration, public APIs, or tool-facing
contracts. This phase is about generator/runtime truth only: supported canonical requests must run
through the real `op_flash_attn_ext` backend route, and unsupported requests must behave
deterministically without silently claiming flash execution.

</domain>

<decisions>
## Implementation Decisions

### Flash Selection Boundary
- **D-01:** Phase 11 only targets the canonical CPU-hosted Llama-68M generation slice already
  shipped by the generator, not broader model or shape coverage.
- **D-02:** Flash adoption happens inside `src/emel/generator/detail.hpp` at the existing
  attention seam after Q/K/V matmuls, RoPE, and K/V cache writes; Boost.SML orchestration stays
  unchanged.

### Unsupported Request Behavior
- **D-03:** If the generator cannot form a canonical `op_flash_attn_ext` request from its current
  runtime state, the request must fail explicitly instead of silently labeling the old
  materialized attention path as flash.
- **D-04:** Broader "non-flash fallback" policy is deferred. Phase 11 should keep behavior
  truthful and narrow rather than introducing a hidden runtime branch that obscures whether flash
  actually ran.

### Proof Of Flash Adoption
- **D-05:** Generator-owned observability must distinguish flash-attention dispatch from generic
  kernel matmul traffic so automated tests and later parity work can prove flash selection.
- **D-06:** Phase 11 proof stays local to generator/runtime tests and state inspection. Parity and
  benchmark publication remain deferred to Phases 12 and 13.

### Operand And Cache Contract
- **D-07:** The generator must feed `op_flash_attn_ext` the same position-major key/value cache
  layout it already owns today rather than introducing a new cache format for this phase.
- **D-08:** No new public C API, CLI flag, or runtime configuration knob is introduced for flash
  adoption; selection is derived from the existing canonical runtime context.

### the agent's Discretion
- Exact generator-local observability fields or counters needed to prove flash dispatch.
- Exact helper decomposition for building tensor views over the existing K/V caches.
- Whether now-unused materialized-attention scratch buffers are removed in Phase 11 or left for a
  later cleanup, as long as flash-path truth and testability are preserved.

</decisions>

<specifics>
## Specific Ideas

- Build the flash request from the existing `backend.q`, `backend.key_cache`, `backend.value_cache`,
  and `backend.attn_ctx` buffers immediately after cache population in `run_layer(...)`.
- Reuse the generator's existing `emel::kernel::sm` member so the adopted path still travels
  through the real generator -> graph -> processor -> kernel chain.
- Add flash-specific dispatch accounting alongside `kernel_dispatch_calls` so Phase 12 can expose
  proof of flash execution without inventing a new tool-only seam.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Scope And Rules
- `.planning/ROADMAP.md` — Phase 11 goal, success criteria, and phase ordering inside v1.2.
- `.planning/REQUIREMENTS.md` — `GEN-01` and `GEN-02`, plus later parity/benchmark boundaries.
- `AGENTS.md` — project engineering contract, including SML, performance, and flash-attention
  truthfulness constraints.
- `docs/rules/sml.rules.md` — RTC/no-queue semantics and state-machine constraints that must remain
  intact.

### Prior Phase Outputs
- `.planning/phases/10-flash-kernel-bring-up/10-CONTEXT.md` — locked scope from the kernel
  bring-up phase.
- `.planning/phases/10-flash-kernel-bring-up/10-RESEARCH.md` — shared-kernel rationale and the
  canonical flash contract already implemented.
- `.planning/phases/10-flash-kernel-bring-up/10-VERIFICATION.md` — verified backend/kernel truths
  Phase 11 is allowed to build on.

### Generator And Kernel Seams
- `src/emel/generator/detail.hpp` — current materialized attention path, cache layout, and
  generator/kernel integration seam.
- `src/emel/generator/context.hpp` — generator-owned runtime state and backend storage.
- `src/emel/generator/actions.hpp` — generator runtime orchestration around graph execution.
- `src/emel/generator/sm.hpp` — existing state-machine surface and observability getters.
- `src/emel/kernel/detail.hpp` — canonical `op_flash_attn_ext` contract and workspace-backed
  implementation from Phase 10.
- `src/emel/kernel/events.hpp` — flash-attention kernel event type and tensor view shapes.

### Existing Proof Surfaces
- `tests/generator/lifecycle_tests.cpp` — current generator-runtime behavioral coverage.
- `tools/paritychecker/parity_runner.cpp` — later parity surface that still uses
  `compute_attention(...)` directly today and will need truthful flash proof in Phase 12.
- `tools/paritychecker/paritychecker_tests.cpp` — current assertions around generator kernel
  dispatch observability.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/generator/detail.hpp`: `make_src_view(...)` and `make_dst_view(...)` already build
  kernel tensor views for generator-owned buffers.
- `src/emel/generator/detail.hpp`: `native_backend` already owns `emel::kernel::sm kernel`,
  `kernel_kind`, and `kernel_dispatch_calls`, so flash dispatch can stay in the existing runtime.
- `src/emel/kernel/detail.hpp`: `run_flash_attn_ext_with_workspace(...)` already validates the
  canonical contract and reuses backend-owned workspace.

### Established Patterns
- Generator matmuls already dispatch through `backend.kernel.process_event(...)`; flash adoption
  should use the same kernel actor rather than bypassing it with a helper-only shortcut.
- `run_layer(...)` currently computes Q/K/V, applies RoPE, copies K/V into persistent caches, then
  materializes attention via `compute_attention(...)` before the attention-output matmul.
- Existing parity tooling only proves generic kernel dispatch counts today, not flash-specific
  dispatch, so Phase 11 needs more precise internal observability before Phase 12 can publish
  truthful proof.

### Integration Points
- The direct insertion point is `src/emel/generator/detail.hpp::run_layer(...)`, replacing the
  `compute_attention(...)` call with `op_flash_attn_ext` request construction and dispatch.
- Generator tests can extend `tests/generator/lifecycle_tests.cpp` and `src/emel/generator/sm.hpp`
  getters to assert that canonical generation used flash dispatch rather than only generic kernel
  work.
- Phase 12 parity work will likely consume the same generator-local observability added here, so
  keep it durable and non-tool-local.

</code_context>

<deferred>
## Deferred Ideas

- User-visible or paritychecker-visible proof that flash executed belongs to Phase 12.
- Benchmark compare rows and maintained artifact updates belong to Phase 13.
- Broader non-canonical fallback policy, multi-model rollout, and backend-specific flash tuning are
  out of scope for Phase 11.

</deferred>

---
*Phase: 11-generator-flash-adoption*
*Context gathered: 2026-03-21*
