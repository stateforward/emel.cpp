# Phase 10: Flash Kernel Bring-Up - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

## Phase Boundary

Phase 10 delivers the canonical EMEL-owned `op_flash_attn_ext` path in `src/emel/kernel` for the
CPU-hosted Llama-68M slice, plus persistent workspace semantics that avoid hot-path allocation.
This phase does not widen model scope, does not adopt the operator in the shipped generator flow
yet, and does not claim user-visible parity or benchmark completion.

## Implementation Decisions

### Supported Kernel Scope
- Phase 10 stays canonical-only.
- The first supported contract is the exact causal Llama-68M attention shape needed by the shipped
  slice, not a broader family of compatible shapes.
- Generic or near-generic flash-attention coverage is explicitly deferred.

### Unsupported Request Behavior
- Unsupported or non-canonical flash-attention requests must reject explicitly.
- Phase 10 must not silently fall back to the old materialized attention path under a flash label.
- Fallback policy beyond explicit rejection is a later-phase decision.

### Proof Of Completion
- Kernel-local proof is sufficient for Phase 10.
- Required proof is shared-kernel correctness plus evidence that persistent workspace/buffers are
  reused without hot-path allocation churn.
- User-visible dump or parity evidence is deferred until generator adoption and parity phases.

### Claude's Discretion
- Exact scalar/shared implementation structure inside `src/emel/kernel`.
- Exact test shape fixtures, as long as they stay within the canonical Llama-68M contract.
- Exact persistent-workspace ownership shape, as long as it remains reusable and does not broaden
  API/runtime boundaries.

## Specific Ideas

- Keep Boost.SML orchestration unchanged; this phase is a data-plane replacement only.
- Treat "flash attention" claims narrowly until the shipped generator path actually adopts the new
  kernel in Phase 11.
- Avoid proving completion through `tools/` output this early; Phase 10 should finish with kernel
  truth, not premature operator-surface claims.

## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/events.hpp`: already declares `op_flash_attn_ext` and dispatch scaffolding.
- `src/emel/kernel/*/sm.hpp`: backend state machines already have dispatch rows for
  `dispatch_op_flash_attn_ext`.
- `src/emel/kernel/detail.hpp`: existing shared-kernel detail surface is the narrowest place to
  land canonical scalar correctness before backend-specific optimization.

### Established Patterns
- `docs/rules/sml.rules.md` and `AGENTS.md` already lock this work into RTC, no-queue,
  bounded-action behavior.
- The repo roadmap already fixes phase order: kernel bring-up first, generator adoption second,
  parity third, benchmark evidence fourth.
- Flash-attention claims must stay truthful: no silent fallback, no tool-local compute substitute,
  no parity claims without aligned execution.

### Integration Points
- `src/emel/generator/detail.hpp` still uses `compute_attention(...)` and is the later Phase 11
  adoption seam, not the Phase 10 delivery target.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` currently force
  reference flash attention off; that alignment work belongs after kernel bring-up.
- Current shipped benchmark/parity surfaces already exist and should remain unchanged during this
  phase except where kernel-only tests need coverage.

## Deferred Ideas

- Generator routing through flash attention belongs to Phase 11.
- User-visible proof that the flash path executed belongs to Phase 12.
- Reference alignment and benchmark evidence belong to Phases 12 and 13.
- Broader shape support, backend-specific optimization, and non-canonical model rollout are out of
  scope for Phase 10.

---
*Phase: 10-flash-kernel-bring-up*
*Context gathered: 2026-03-13*
