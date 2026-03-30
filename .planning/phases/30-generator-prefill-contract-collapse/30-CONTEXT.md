---
phase: 30
slug: generator-prefill-contract-collapse
created: 2026-03-29
status: ready
---

# Phase 30 Context

## Phase Boundary

Collapse the top-level generator prefill routing matrix into explicit request-scoped prefill
contracts and shared compute/result states. This phase does not extract a child machine yet.

## Implementation Decisions

### Prefill Contract
- Prefill route choice must resolve into an explicit request-scoped contract carried on the
  generate runtime event, not generator context.
- Contract choice stays modeled through explicit guards and transitions; actions may only stamp a
  compile-time contract constant, never branch to choose one.
- The contract must encode the actual prefill behavior dimensions currently duplicated in the
  table: flash/nonflash, materialized/preselected, scalar/chunk4.

### Top-Level Collapse
- The parent `generator::sm` keeps prefill in-domain for this phase.
- The duplicated prefill success/error fan-out must collapse around shared compute/result states.
- Decode routing after prefill success must still remain explicit: materialized contracts flow to
  decode selection, preselected contracts flow to decode-preselected.

### Guardrails
- No new generator context fields for request-scoped orchestration.
- No hidden route selection in action/detail helpers.
- Decode extraction and `sm_any` attention-family work remain out of scope for this phase.

## Existing Code Insights

### Duplication Source
- `src/emel/generator/sm.hpp` currently duplicates eight prefill compute branches plus repeated
  success/error result handling.
- `src/emel/generator/actions.hpp` already has compile-time-specific request helpers for each
  concrete prefill compute contract.
- `src/emel/generator/guards.hpp` already exposes the runtime capability predicates needed to
  resolve those contracts explicitly.

### Likely Reuse
- A request-scoped enum on `event::generate_ctx` is acceptable because it stays on the runtime
  event rather than generator context.
- Compile-time action templates can stamp a constant contract without introducing runtime control
  flow.

## Specific Ideas

- Introduce one `prefill_compute_contract` enum plus constant-stamping actions.
- Replace the old `prefill_compute_*` state family with:
  - runtime decision
  - flash/nonflash decision
  - shared prefill compute result decision
- Keep the concrete compute request actions, but make them feed one shared result decision.
