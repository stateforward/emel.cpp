---
title: graph architecture design
status: draft
---

# graph architecture design

this document defines graph. the graph is an actor (sm) that acts as the compute
manager for the generator. it owns the DAG topology — the nodes in topological
order and the edges (source mappings).

## role
- orchestrate the compute pipeline: delegate DAG construction to assembler, and
  execution to processor.
- own the DAG topology and the Data-Oriented Design (DOD) tensor arrays: nodes
  in execution order, edges (source indices per node), and flat tensor metadata
  (`refs[]`, `states[]`, `pointers[]`).
- provide traversal order for graph/processor.
- manage tensor lifecycles procedurally: set `refs` during bind, allowing the
  processor to decrement them during execution without SML event overhead.

## composition
- owned by generator.
- owns graph/assembler (constructs or reuses the DAG).
- owns graph/processor (executes the DAG via kernel).
- owns the flat DOD tensor arrays (populated by assembler during reserve, managed
  here).

## state model (draft)

```text
uninitialized ──► reserved ──► idle ──► executing ──► done
                                 ▲                      │
                                 └──────────────────────┘
```

- `uninitialized` — no graph built.
- `reserved` — assembler has constructed worst-case DAG and allocated tensor
  buffers. ready for real assembles.
- `idle` — graph ready, waiting for a compute request.
- `executing` — processor is procedurally walking nodes and dispatching ops to kernel.
- `done` — execution complete, results in output tensors. transitions back
  to `idle`.

## events (draft)
- `event::reserve` — from generator at init. forwards to assembler. inputs:
  model metadata (worst-case dims). transitions: `uninitialized → reserved`.
- `event::compute` — from generator per ubatch. inputs: `batch::plan`.
  forwards to assembler for assemble/reuse, then to processor for execution.
  transitions: `idle → executing`.
- `events::compute_done` — execution complete. outputs: output tensor
  pointers (logits). transitions: `executing → done → idle`.
- `events::compute_error` — error. outputs: error_out.

## DAG topology (DOD layout)
- `nodes[]` — tensor indices in topological (execution) order.
- `edges[]` — per node, source tensor indices (up to 4 sources: src0, src1,
  src2, src3). index into the DOD arrays.
- `tensor_refs[]` — flat array of active consumer counts.
- `tensor_states[]` — flat array of tensor lifecycle states (`allocated`, `empty`, `filled`).
- `tensor_pointers[]` — flat array of hardware buffer pointers.

## ref management
- on bind: graph walks `nodes[]`, counts consumers for each tensor
  (how many nodes list it as a source), and initializes `tensor_refs[id]` with the
  consumer count.
- on op completion: the `graph/processor` directly decrements `tensor_refs[src_id]`.
  when a tensor's refs hit zero, its state returns to `empty` — the buffer region
  is available for reuse by the next tenant.

## relationship to assembler and processor
- assembler constructs/modifies the DAG topology and DOD arrays. the graph owns
  the topology after assemble.
- processor reads the topology from the graph, walks nodes in a fast `for` loop,
  and uses the Opcode Router to dispatch each node's op to the kernel. after each
  op, it procedurally decrements `refs` on the sources.
