---
title: memory/hybrid architecture design
status: draft
---

# memory/hybrid architecture design

this document defines the hybrid memory actor. it provides a unified lifecycle surface for models
that utilize both KV Cache and Recurrent memory (e.g., Jamba, certain RWKV variants).

## role
- act as a transparent orchestrator over both a `memory/kv::sm` and a `memory/recurrent::sm`.
- expose a single `memory` API to the `generator`.
- synchronize lifecycle events across both underlying memory architectures.

## architecture: the unified facade
rather than building a complex, three-tiered "coordinator" state machine, the hybrid memory actor
is a simple facade. the true complexity of PagedAttention and Recurrent State copying remains
isolated inside their respective actors.

when the `generator` issues a lifecycle event, the hybrid actor simply multi-casts it:

1. **allocate:** dispatches to both `kv` and `recurrent`. if *either* fails (hits `out_of_memory`),
   the hybrid actor gracefully rolls back the successful one and returns `out_of_memory` to the
   generator.
2. **branch:** dispatches to both. `kv` handles the blazing-fast DOD reference bump for zero-copy
   block sharing, while `recurrent` handles the physical state buffer copy into a new slot.
3. **free:** dispatches to both, freeing the recurrent slot and dropping the KV block references via
   their respective DOD arrays.

## composition
- owned by the `generator`.
- owns one instance of `memory/kv::sm`.
- owns one instance of `memory/recurrent::sm`.

## responsibilities
- multi-cast sequence lifecycle events (`allocate`, `branch`, `free`) to both sub-actors.
- deterministic error handling: if one subsystem fails an allocation, ensure the other is safely
  rolled back to maintain sequence parity between the two memory domains.
- provide a unified `memory::any` view for the `graph/processor` to bind during execution.
