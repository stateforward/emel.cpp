# memory/coordinator architecture design (rolling)

this document captures the intended architecture for the memory/coordinator actor cluster and its
relationship to generator orchestration.

## role
- memory/coordinator is the lifecycle orchestrator for memory modules (kv, recurrent, hybrid).
- it owns prepare/update/full phases, status mapping, and error translation.
- it is a pure orchestration layer; underlying memory actors do the real work.
- memory/coordinator is an `sm_any` wrapper that selects a concrete coordinator variant.

## variant docs
- `memory/coordinator/kv.design.md`
- `memory/coordinator/recurrent.design.md`
- `memory/coordinator/hybrid.design.md`

## current state (implementation)
- `memory/coordinator` implements the generic phase machine and status publishing.
- `memory/coordinator/kv` implements the same phase shape with a policy-only context
  (no direct kv/cache calls yet).
- `memory/coordinator/recurrent` and `memory/coordinator/hybrid` exist with similar phase shape.

## intended ownership (inference pipeline)
- generator is the least common owner for memory coordination.
- generator owns a memory/coordinator instance and drives:
  - `prepare_update` (shift/copy/update readiness),
  - `prepare_full` (worst-case graph sizing),
  - `prepare_batch` (batch-specific memory preparation).

## boundary with kv/cache
- kv/cache is the memory actor for attention cache semantics (slots, apply, rollback, seq ops).
- memory/coordinator/kv is the gateway to kv/cache lifecycle phases.

## open questions
- should `memory/coordinator/kv` own kv/cache directly or accept an injected handle?
- how are `failed_prepare` vs `failed_update` mapped into EMEL errors?

## next steps
- decide if `memory/coordinator/kv` will dispatch kv/cache events directly.
