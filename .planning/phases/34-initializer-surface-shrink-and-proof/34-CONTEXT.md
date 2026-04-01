---
phase: 34
slug: initializer-surface-shrink-and-proof
created: 2026-03-31
status: ready
---

# Phase 34 Context

## Phase Boundary

Close the initializer milestone by proving the reduced parent generator surface, refreshing the
generated architecture docs, and rerunning the maintained generator/parity/benchmark lanes.

## Implementation Decisions

### Parent Surface
- Phase 33 already removed the inline initialize pipeline from the parent generator table.
- Phase 34 should avoid introducing any new machines or broader lifecycle redesign.
- Remaining work is allowed to be documentation, test, and proof oriented if the parent surface is
  already materially smaller.

### Proof Scope
- Keep proof focused on maintained generator surfaces touched by the initializer extraction.
- Preserve Llama and canonical Qwen generator behavior.
- Treat benchmark and parity proof as milestone-close checks, not as an excuse to broaden the
  refactor.

## Guardrails

- No decode child extraction.
- No attention-family `sm_any` split.
- No generator public API changes.
