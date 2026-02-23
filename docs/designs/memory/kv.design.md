---
title: memory/kv architecture design
status: draft
---

# memory/kv architecture design

this document defines the kv cache actor. it acts as the central paged block manager, orchestrating
allocations, prefix caching, and zero-copy sequence branching using a strict Data-Oriented Design
(DOD) page table.

## role
- own the physical kv tensor buffers and logically partition them into fixed-size blocks (e.g., 16
  or 32 tokens).
- manage the "page table" (sequence-to-block mappings) and block reference counts.
- orchestrate complex sequence lifecycle events (allocate, free, branch) received from the `generator`.
- handle out-of-memory states safely via SML transitions rather than crashing or returning silent
  errors.

## architecture shift: DOD with `sml::sm_pool`
initially, every block of the kv cache was modeled as an independent `boost::sml` actor to track references.
however, dispatching SML events (like `event::link` or `event::unlink`) to thousands of independent blocks
introduced unacceptable overhead in the hot path.

to achieve maximum performance without abandoning the safety of the Actor Model, the `memory/kv` actor manages
its blocks using `boost::sml::sm_pool<block_sm>`.
- **cache locality:** `sm_pool` lays out the state machine data contiguously in memory.
- **batch dispatch:** when a sequence is freed or branched, the `memory/kv` actor does not loop over individual
  actors. instead, it calls the highly optimized `process_event_batch(seq_to_blocks, event::unlink{})` API.
- **the result:** this batch API provides an 81% overhead reduction compared to independent actors, bringing the
  cost of SML dispatch so close to raw C arrays (~11% overhead) that we can retain the strict declarative reference
  counting (`empty` -> `filled` -> `empty`) for every single block without compromising performance.

## composition
- owned by the `generator` (or `memory/hybrid` if mixed).
- owns the physical backing tensors for keys and values.

## state model

```text
uninitialized ──► initializing ──► ready
                                     │
ready ──► allocating_sequence ──► (ready | out_of_memory)
  ▲                                  │
  └──────────────────────────────────┘

ready ──► branching_sequence ──► ready
ready ──► freeing_sequence   ──► ready
```

- `uninitialized` — awaiting initial setup.
- `initializing` — partitioning the backing tensors and populating the `free_pool`.
- `ready` — waiting for lifecycle commands.
- `allocating_sequence` — reserving a sequence ID slot. if successful, returns to `ready`. if the
  maximum number of active sequences is reached, drops to `out_of_memory`.
- `branching_sequence` — zero-copy duplication of a parent's block mapping to a new child.
- `freeing_sequence` — unlinking all blocks associated with a sequence ID.
- unexpected events route to `unexpected`.

## responsibilities
- **allocation (`event::allocate_slots`):** when a sequence needs to write new tokens, the manager
  pops `N` indices from the `free_pool`, sets their `block_refs = 1`, and appends them to the
  sequence's mapping array. if the `free_pool` is empty, gracefully reject via SML guard.
- **zero-copy branching (`event::branch_sequence`):** given a parent sequence, duplicate its
  `seq_to_blocks` mapping to a new child sequence. iterate over those blocks and increment
  `block_refs[idx]++`. no physical memory is copied.
- **freeing (`event::free_sequence`):** look up a sequence's blocks. iterate over them and decrement
  `block_refs[idx]--`. if a block's ref count hits 0, push its index back to the `free_pool`. clear
  the sequence mapping.
