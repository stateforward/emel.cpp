# kernel/any architecture design (draft)

this document defines kernel/any. it is the top-level kernel orchestrator that receives
graph scheduling requests from graph/processor and delegates to `kernel::device::any`.

## role
- receive `event::schedule` from graph/processor.
- build or reuse `kernel::instructions` (execution plan + scratch metadata).
- delegate graph computation to the owned `kernel::device::any`.

## composition
- owns `kernel::device::any` (device-level variant dispatch).
- owns `kernel::instructions` (execution plan data).

## events (draft)
- `event::schedule` inputs: bound `graph`, kernel execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- build or reuse `kernel::instructions` for the graph signature + policy.
- dispatch `event::schedule` to `kernel::device::any`.
- ensure device synchronization before returning done.
