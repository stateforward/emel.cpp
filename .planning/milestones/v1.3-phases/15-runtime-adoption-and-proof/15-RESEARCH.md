# Phase 15: Runtime Adoption And Proof - Research

**Date:** 2026-03-22
**Status:** Complete

## Key Findings

1. `src/emel/generator/detail.hpp` already routes the canonical generation layer path through
   `make_flash_attn_request(...)` and `dispatch_flash_attention(...)`, so runtime adoption is
   functionally in place; the missing proof is backend attribution, not a missing generator call
   seam.
2. `src/emel/generator/sm.hpp` already exposes generator-level `generation_kernel_kind()`,
   `generation_kernel_dispatch_calls()`, and `generation_flash_attention_dispatch_calls()`, but it
   cannot currently distinguish optimized AArch64 flash execution from a shared fallback.
3. `src/emel/kernel/aarch64/context.hpp` now stores
   `optimized_flash_dispatch_count` and `shared_flash_dispatch_count`, but those counters are only
   visible inside the backend machine today.
4. `src/emel/kernel/any.hpp` already uses `sm_any::visit(...)`, so it can surface backend-specific
   observability without changing transition tables or introducing new events.
5. `tools/paritychecker/parity_runner.cpp` already emits stable generation proof comments for the
   canonical fixture. Extending those lines with optimized/shared ARM counts is the smallest path
   to satisfying `PAR-03`.
6. `tests/generator/lifecycle_tests.cpp` already contains the right positive and negative runtime
   cases. Phase 15 can extend those tests instead of inventing a new harness.

## Constraints

- `AGENTS.md` forbids state-machine structure changes without asking the user. Phase 15 must stay
  at wrapper/helper/test level only.
- Snapshot-bearing benchmark publication is approval-gated and belongs to Phase 16, not this
  phase.
- Proof must remain portable: on non-AArch64 builds the new counters should remain explicit and
  benign instead of creating false ARM-only failures.

## Chosen Direction

- Surface optimized/shared flash counts from the active kernel wrapper through
  `src/emel/kernel/any.hpp` into `src/emel/generator/sm.hpp`.
- Extend generator lifecycle proof to distinguish canonical ARM optimized execution from negative
  runtime cases.
- Extend paritychecker generation output and tests to publish and verify the optimized/shared
  counts, enforcing ARM-specific truth only when `generation_kernel_kind()` is `aarch64`.

---
*Phase: 15-runtime-adoption-and-proof*
*Research completed: 2026-03-22*
