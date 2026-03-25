# Phase 15: Runtime Adoption And Proof - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 15 proves that the shipped canonical ARM runtime reaches the optimized AArch64
`op_flash_attn_ext` backend path through the existing generator -> graph -> processor -> kernel
chain. This phase stays inside runtime observability and proof surfaces: it does not rewrite SML
structure, widen API surface, or publish new benchmark snapshot artifacts.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Keep all routing and orchestration structure unchanged; only add observability or proof surfaces
  around the already-shipped runtime seam.
- Reuse generator-owned counters and maintained paritychecker output instead of introducing
  tool-local-only runtime counters.
- Defer any checked-in benchmark snapshot or docs publication changes to Phase 16 because
  `AGENTS.md` requires explicit user consent before snapshot updates.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/generator/detail.hpp` already dispatches canonical attention through
  `emel::kernel::event::op_flash_attn_ext` and records generator-level
  `flash_attention_dispatch_calls`.
- `src/emel/generator/sm.hpp` already exposes runtime-facing kernel kind and flash-dispatch
  counters for the shipped generator seam.
- `tools/paritychecker/parity_runner.cpp` already publishes `kernel_dispatch:` and
  `flash_dispatch:` evidence for the canonical generation fixture.
- `tests/generator/lifecycle_tests.cpp` already proves canonical generation success and
  deterministic non-canonical failure without false flash claims.

### Established Patterns
- Runtime proof belongs on shipped seams (`src/` generator/kernal wrappers and maintained tool
  surfaces), not in ad hoc local scripts.
- Backend selection evidence can be exposed through machine wrappers without changing SML
  transition structure.
- Unsupported behavior is considered correct only when it remains explicit and deterministic.

### Integration Points
- `src/emel/kernel/aarch64/sm.hpp` and `src/emel/kernel/any.hpp` are the narrowest seams for
  surfacing backend-specific optimized/shared flash counters.
- `src/emel/generator/sm.hpp` is the shipped public C++ seam that paritychecker and tests already
  use for runtime observability.
- `tests/generator/lifecycle_tests.cpp` and `tools/paritychecker/paritychecker_tests.cpp` are the
  current proof surfaces for canonical runtime behavior.
- `tools/paritychecker/parity_runner.cpp` is the maintained publication surface for `PAR-03`.

</code_context>

<specifics>
## Specific Ideas

- Add runtime-visible optimized/shared flash counters that stay zero off the AArch64 backend and
  reflect real backend execution when the active kernel is AArch64.
- Extend canonical generation proof to assert optimized-path selection on ARM and zero optimized
  claims for negative or out-of-contract runtime cases.
- Publish the optimized/shared counts in paritychecker output so ARM proof is visible on the
  maintained tool surface before benchmark publication begins.

</specifics>

<deferred>
## Deferred Ideas

- Updating `snapshots/bench/benchmarks_compare.txt` or `docs/benchmarks.md` remains Phase 16 work
  and stays approval-gated.

</deferred>

---
*Phase: 15-runtime-adoption-and-proof*
*Context gathered: 2026-03-22*
