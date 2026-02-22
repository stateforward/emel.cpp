# graph/builder architecture design (draft)

this document defines graph/builder. it builds or reuses an executable graph for a batch plan.

## role
- build or reuse a graph for a specific `batch::plan`.
- output an executable graph instance for graph/processor.

## events (draft)
- `event::build` inputs: `batch::plan` (shape summary + output requirements), `memory::any`
  (prepared view for graph inputs), graph policy (reuse/prepare/reserve flags). model metadata
  and kernel context are injected at construction.
- `events::build_done` outputs: `graph` (executable graph + scheduling metadata), status.
- `events::build_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `building` -> `build_decision` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- compute graph parameters from `batch::plan` + `memory::any`.
- decide reuse vs rebuild.
- reserve/allocate graph buffers as needed.
