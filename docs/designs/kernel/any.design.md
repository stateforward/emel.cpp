# kernel/any architecture design (draft)

this document defines kernel/any. it is the top-level kernel orchestrator that receives
graph scheduling requests from graph/processor and delegates to `kernel::device::any`.

## role
- receive `event::schedule` from graph/processor.
- walk graph nodes and dispatch per-node opcode events to `kernel::device::any`.
- bridge runtime opcode IDs to compile-time event types via `make_dispatch_table`.

## composition
- owns `kernel::device::any` (device-level variant dispatch).

## events (draft)
- see `kernel/events.design.md` for full event catalog.
- `event::schedule` inputs: bound `graph`, kernel execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.
- per-node: `op::*` events dispatched to device via `make_dispatch_table`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- validate opcode ID range before indexing the dispatch table.
- walk graph nodes in topological order.
- dispatch one `op::*` event per node to `kernel::device::any`.
- propagate device errors back as `events::schedule_error`.
- ensure device synchronization before returning done.
