---
title: tensor architecture design
status: draft
---

# tensor architecture design

this document defines the tensor architecture. a tensor models a buffer with reference-counted
consumers, analogous to a filesystem inode with hardlinks.

## role
- hold a pointer to a data buffer and track its lifecycle: write, read, clear.
- track consumer references (refs). when refs hit zero, the tensor returns to empty — data is
  logically gone, ready for the next cycle.
- know its operation (what produced it) and its shape/stride metadata.

## architecture shift: benchmark-driven DOD (`sml::sm_pool`)
initially, every tensor was modeled as an independent `boost::sml` actor. however, benchmarking
revealed that dispatching SML events to tens of thousands of individual actors in a random-access
pattern (typical of walking a sparse DAG) resulted in a massive performance penalty compared to
flat arrays (~120k ns vs ~21k ns).

to achieve maximum performance in the execution hot path without abandoning the safety and
declarative nature of the Actor Model, tensors are managed using `boost::sml::sm_pool`.

- **`sm_pool`:** the `graph/graph` (or a dedicated topology struct) owns an `sml::sm_pool<tensor_sm>`.
  this lays out the state machine data contiguously in memory (Data-Oriented Design).
- **Batch Dispatch:** during the `graph/processor` loop, instead of dispatching to individual actors,
  the processor uses the new `process_event_batch(indices, event)` API.
- **The Result:** benchmarking shows `sm_pool` batch dispatch is **5.1x faster** than independent
  SML actors and incurs only an **~11% overhead** compared to raw procedural C arrays. we retain the
  strict "inode" SML safety guarantees while hitting raw hardware memory throughput.

## inode analogy
- buffer pointer = block pointer on disk.
- refs = hardlinks / open file handles.
- reserve = allocate inode, assign block pointer.
- bind = `touch` — file exists, empty.
- kernel executes op = `write` — data written to blocks.
- consumer references = hardlinks to the inode.
- consumer op completes = `unlink` — decrement refs.
- refs == 0 = blocks reclaimable — tensor returns to empty.

## state model (logical)

```text
unallocated ──► allocated ──► empty ──► filled
                                ▲          │
                                │   refs++ / refs--
                                │          │
                                └──────────┘
                                 refs == 0
```

- `unallocated` — no buffer. tensor knows its shape, op, and sources but has no storage.
- `allocated` — buffer pointer assigned (during reserve). one-time transition.
- `empty` — buffer pointer valid, no live data. ready to be written by the kernel.
- `filled` — op completed, data live in buffer. consumers read from it. refs tracks active consumers.
  when refs == 0, transitions back to `empty`.

per execution cycle, the tensor logically moves `empty → filled → empty`.

## tensor types
- **compute tensors** — produced by an op. follow the full `empty → filled → empty` cycle.
- **leaf tensors (weights)** — externally provided data. go to `filled` and stay there permanently.
  refs never reach zero.
- **input tensors** — per-batch input data. written externally before execution, follow the normal cycle.

## refs mechanics
- refs are set during graph bind (entry to `empty`) based on the number of consumer tensors that
  source from this tensor.
- when a consumer tensor's op completes, the `graph/processor` directly decrements the ref count
  of each source tensor.
- if a ref count hits 0, the source tensor's state is set back to `empty`.

## buffer ownership
- a tensor does not own its raw allocation. it holds a pointer to a region in a shared buffer,
  assigned during reserve by the allocator.
- multiple tensors with non-overlapping lifetimes share the same physical memory.
- the tensor's contract: when filled, the region is valid. when empty, it does not touch it.
