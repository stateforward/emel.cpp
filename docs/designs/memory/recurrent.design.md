---
title: memory/recurrent architecture design
status: draft
---

# memory/recurrent architecture design

this document defines the recurrent memory actor. it provides storage and lifecycle management for
models that maintain a fixed-size state vector per sequence (e.g., RWKV, Mamba, Jamba).

## role
- manage recurrent state buffers (which are fixed-size per sequence, unlike kv caches that grow per
  token).
- orchestrate sequence lifecycle events (allocate, free, branch) received from the `generator`.
- enforce capacity boundaries safely via SML transitions.

## architecture shift: DOD with `sml::sm_pool`
unlike the kv cache which partitions memory into paged blocks, recurrent memory is much simpler.
each sequence requires a single, contiguous, fixed-size state buffer.

to maximize cache locality and execution speed while preserving Actor Model safety, the recurrent states
are managed using `boost::sml::sm_pool<recurrent_slot_sm>`.
- **cache locality:** `sm_pool` ensures all slot actors are contiguous in memory.
- **declarative safety:** each slot is a true SML actor tracking its lifecycle (`empty` -> `filled`).
- **batch dispatch:** the `memory/recurrent` manager coordinates slots using the high-performance
  `process_event_batch` API, ensuring allocation and freeing remain strictly safe with only ~11% overhead
  compared to raw procedural arrays.

1. **fixed allocation:** allocating a sequence simply finds an inactive slot in the array and marks
   it active.
2. **explicit branching:** because recurrent state is mutated at every step (unlike historical kv
   keys/values which are read-only), sequence branching requires a physical memory copy. when sequence
   B branches from sequence A, the recurrent manager finds a new inactive slot for B, and physically
   copies the data from A's slot into B's newly allocated slot.

## composition
- owned by the `generator` (or `memory/hybrid` if mixed).
- owns the physical backing tensors for recurrent states.

## state model

```text
uninitialized ──► initializing ──► ready
                                     │
ready ──► allocating_sequence ──► (ready | out_of_memory)
  ▲                                  │
  └──────────────────────────────────┘

ready ──► branching_sequence ──► (ready | out_of_memory)
ready ──► freeing_sequence   ──► ready
```

- `uninitialized` — awaiting initial setup.
- `initializing` — partitioning the backing tensor into fixed-size sequence slots.
- `ready` — waiting for lifecycle commands.
- `allocating_sequence` — finding an empty state slot. if capacity is reached, gracefully reject
  via SML guard (`out_of_memory`).
- `branching_sequence` — allocating a new slot and copying the parent's state data into it.
- `freeing_sequence` — clearing a sequence ID mapping and marking the slot inactive.
- unexpected events route to `unexpected`.

## responsibilities
- **initialization:** partition the backing tensor into fixed-size sequence slots based on model dims.
- **allocation:** map a requesting sequence ID to the first available `slot_active == false` index.
- **branching:** duplicate the parent sequence's physical state data into a new child sequence's slot.
- **freeing:** clear the sequence mapping, making the physical slot available for future allocations.
