# graph/processor architecture design (draft)

this document defines graph/processor. it binds a prepared graph for one plan step and dispatches
kernel execution.

## role
- bind a `graph` for a plan step, dispatch kernel execution, and extract outputs.

## events (draft)
- `event::process` inputs: `graph` (built by graph/builder), `batch::plan` (token batch pointers +
  step mapping), `memory::any` (prepared view), output buffers for logits or pooled embeddings.
- `events::process_done` outputs: produced row counts + row mapping, status.
- `events::process_error` outputs: error_out.

## state model (draft)
- `idle` -> `binding_inputs` -> `running` -> `extracting` -> `done`.
- failures route to `errored`, unexpected events to `unexpected`.

## responsibilities
- bind step inputs into the graph.
- dispatch `event::schedule` to `kernel::any` (kernel schedules and computes subgraphs internally).
- extract outputs into provided buffers.
