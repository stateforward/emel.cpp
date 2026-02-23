---
title: graph/processor architecture design
status: draft
---

# graph/processor architecture design

this document defines graph/processor. it binds a prepared graph for one plan step and dispatches
kernel execution.

## role
- bind a prepared graph topology for one plan step, dispatch kernel execution, and extract outputs.
- act as the runtime-to-compile-time bridge (the "Opcode Router") between the dynamic DAG and the
  strongly-typed `kernel::any` actor.
- procedurally manage tensor lifecycles during the hot loop by decrementing reference counts in the
  graph's DOD arrays as nodes complete.

## architecture shift: the opcode router
the `kernel` domain achieves zero-overhead execution by relying on compile-time typed events (e.g.,
`kernel::op::add{...}`). however, the execution DAG is built dynamically at runtime (e.g., parsing a
GGUF file), meaning a tensor node only holds a runtime enum representing its operation
(e.g., `node->opcode == OP_ADD`).

to bridge this gap, the `graph/processor` acts as the **Opcode Router**. during its execution walk,
it reads the runtime `node->opcode` enum and uses `sml::utility::make_dispatch_table` (or a static
`switch` statement) to construct the corresponding strongly-typed `op::*` event and dispatch it to
`kernel::any`. this isolates the dynamic-to-static transition to a single, highly optimized location
in the hot path.

## composition
- owned by graph::sm.

## events (draft)
- `event::process`
  - inputs: bound DAG topology (from graph::sm context), `batch::plan` (token batch pointers +
    step mapping), `memory::any` (prepared view), output buffers for logits or pooled embeddings,
    and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: produced row counts + row mapping in the provided buffers, invoking the appropriate
    callback before returning to prevent caller context reads.

## state model (draft)
- `idle` -> `binding_inputs` -> `running` -> `extracting` -> `done`.
- failures route to `errored`, unexpected events to `unexpected`.

## responsibilities
- bind step inputs into the graph.
- execute a blazing-fast procedural `for` loop over the DAG nodes in topological order.
- use the Opcode Router to synchronously dispatch math operations as `op::*` events to `kernel::any`.
- immediately decrement the `refs` count in the graph's DOD tensor arrays for a node's sources after
  its kernel operation completes.
- extract outputs into provided buffers.
