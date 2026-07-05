# Phase 245 Context: Block Geometry Ownership and Slot Allocation

## Source-backed baseline (verified 2026-07-04)

Already wired (no work needed beyond tests):
- Generator initializer dispatches `memory::event::reserve` with
  `limits.block_capacity` / `limits.block_tokens`
  (`src/emel/text/generator/initializer/actions.hpp:97`).
- Prefill dispatches `allocate_slots` for `prompt_token_count`
  (`src/emel/text/generator/prefill/actions.hpp:12`); decode dispatches
  `allocate_slots` token_count=1 per step (`src/emel/text/generator/actions.hpp:455`),
  each with ok/invalid/backend decision triples in the transition tables.
- kv machine guards check free-pool capacity before exec
  (`memory/kv/guards.hpp: allocate_slots_request_capacity_valid`), and
  `out_of_memory` maps to the generator backend-error channel
  (`text/generator/guards.hpp:72`).

Gaps this phase closes:
1. **Geometry contract (KVM-01):** `detail::prepare` sizes the physical cache from
   model `n_ctx` with no tie to block geometry; `limits` defaults are 0 and the kv
   machine resolves defaults privately (`exec_reserve`), invisible to the generator.
   Constants are duplicated (`view.hpp` MAX_SEQUENCES/MAX_BLOCKS_PER_SEQUENCE vs
   `kv/detail.hpp` copies) and `blocks_for_length` exists twice (kv actions + guards).
2. **Ascending block ids:** `reset_runtime` fills the free stack so pops yield
   DESCENDING ids (first block = max_blocks-1). Fresh single-sequence growth must map
   to ascending contiguous physical spans or the later flash-contiguity guard
   (Phase 248) can never pass.
3. **Exhaustion proof:** no generator-level test drives block exhaustion through the
   public machine surface.

## Design decisions

- Geometry contract home: `src/emel/memory/view.hpp` (already the cross-domain view
  contract; snapshot carries `block_tokens`). Adds DEFAULT_BLOCK_TOKENS, MAX_BLOCKS,
  `blocks_for_tokens`, `positions_capacity_for`. kv/detail constants become aliases.
- Backend consumes geometry at prepare: new `native_backend.kv_block_tokens` and
  `kv_positions_capacity` (= blocks_for_tokens(block_tokens, n_ctx) * block_tokens);
  cache extents/strides switch from `n_ctx` to `kv_positions_capacity`. For
  n_ctx % block_tokens == 0 (all maintained fixtures) this is bit-identical.
- Defaults resolve branch-free in `begin_initialize` (mirrors kv `exec_reserve`
  precedent) so `limits` always holds the effective geometry.
- Coverage validation is a pure initializer guard on the `reserving_memory` decision:
  `blocks_for_tokens(block_tokens, n_ctx) <= block_capacity` else invalid_request.
  No validation branching inside prepare beyond its pre-existing pattern.
- Free stack fills reversed (`free_stack[i] = max_blocks - 1 - i`) so allocation pops
  ascend 0,1,2,...; frees still push LIFO.

## Out of scope for this phase

Snapshot-consuming addressing (247+), flash contiguity guard (248), recurrent slot
(249). No public API changes: `event::initialize` already carries the fields.
