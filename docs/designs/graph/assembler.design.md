---
title: graph/assembler architecture design
status: draft
---

# graph/assembler architecture design

this document defines graph/assembler. it assembles or reuses the DAG topology
and creates tensor actors for graph::sm.

## role
- assemble or reuse an execution DAG for a specific `batch::plan`.
- create tensor actors and assign their static memory offsets.
- construct the mathematical topology (nodes, edges, leafs) and hand it to graph::sm.

## composition
- creates tensor actors on behalf of graph::sm during reserve, reuses them across cycles.
- uses `graph/allocator` to compute static interval memory assignments and assign offsets
  to tensor actors during reserve or topology changes.

## events (draft)
- `event::reserve`
  - inputs: worst-case dimensions (max sequence length, max batch size), model metadata,
    and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: invokes callback upon successfully creating tensor actors and finding worst-case offsets.
    transitions: `uninitialized → reserved`.
- `event::assemble`
  - inputs: `batch::plan` (shape summary + output requirements), `memory::any` (prepared view for graph inputs),
    graph policy (reuse flags), and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
    model metadata and kernel context are injected at construction.
  - outputs: populates caller-provided pointers to the DAG topology and tensor actor references,
    invoking the callback before returning.

## state model (draft)
- `uninitialized` -> `reserving` -> `idle`.
- `idle` -> `assembling` -> `assemble_decision` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- compute graph shape parameters from `batch::plan` + `memory::any`.
- decide reuse vs reassemble (topology matching against existing tensor actors).
- construct tensor DAG from model forward pass operations (calculate tensor dimensions,
  wire source edges into topology, specify `op::*` kernels).
- run `graph/allocator` only if the DAG topology changes from the worst-case watermark.
- support `reserve` from worst-case graph to avoid reallocation during inference.

## relationship
- owned by graph::sm. see `graph/graph.design.md`.
