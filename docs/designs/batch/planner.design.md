---
title: batch/planner architecture design
status: draft
---

# batch/planner architecture design

this document defines batch/planner. it converts a logical, unbounded token batch into a series of
strictly bounded, executable `batch::plan` steps (ubatches) that adhere to the static memory
constraints established by the graph and allocator.

## role
- convert a logical `token::batch` (containing tokens from mixed prefill and decode sequences) into
  one or more `batch::plan` execution steps.
- enforce the worst-case hardware/graph constraints (e.g., max tokens per step, max batch size)
  computed during the `graph/assembler` reserve phase.
- generate flat, zero-copy mapping arrays (positional coordinates, routing IDs, logit targets)
  for the `graph/processor` to consume directly, ensuring zero overhead during execution.

## architecture shift: stateless pure planning
in dynamic systems (like `llama.cpp`), batch planning is tightly entangled with memory allocation,
validation, and data copying within a monolithic class. 

in `emel`, because the `graph/allocator` handles memory statically and the `token::batch` is sanitized
prior to planning, the `batch/planner` is **purely stateless and mathematical**. it does not allocate
memory for tokens or copy token data. it simply computes index offsets, step boundaries, and flat routing
tables. it acts as a declarative bridge between unbounded user requests and strictly bounded graph execution.

## events
- `event::plan`
  - inputs: `token::batch` (sanitized tokens + sequence metadata), graph constraints
    (max batch size, max step tokens), a planning policy (e.g., sequential vs. equal split), and
    optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: executes the planning logic, populates the provided `batch::plan` output buffers, and
    invokes the appropriate callback before returning. completely avoids exposing internal context.

## state model

```text
uninitialized ──► binding ──► idle
                               │
idle ──► planning ──► plan_decision ──► (done | errored)
  ▲                                        │
  └────────────────────────────────────────┘
```

- `uninitialized` — awaiting initial setup.
- `binding` — configuring static constraints from the `graph/assembler` watermark.
- `idle` — waiting for a new token batch.
- `planning` — sorting sequences (prefill vs. decode) and determining step boundaries via pure actions.
- `plan_decision` — evaluating if the generated ubatches fit the constraints and populating mapping tables.
- `done` — planning complete, transitions back to `idle` emitting `events::plan_done`.
- unexpected events route to `unexpected`.

## responsibilities

1. **ubatch slicing (pure actions):**
   - use pure SML actions (like `action::split_sequential` or `action::split_equal`) to slice large
     prefill sequences into smaller steps if they exceed the `max_tokens` watermark.
   - interleave prefill chunks with single-token decodes to maximize compute utilization (continuous batching),
     while strictly respecting the `max_batch_size`.

2. **zero-copy token indexing:**
   - the resulting `batch::plan` does not duplicate token IDs or embeddings. it simply contains integer bounds
     (e.g., `start_idx`, `end_idx`) pointing back to the original `token::batch` array.

3. **generalized positional coordinates:**
   - compute absolute sequence positions (`position_ids`) for each token.
   - the planner is agnostic to the model's positional embedding technique (RoPE, ALiBi, YaRN). it merely
     provides the topological coordinate ("this token is at position 5"). the `graph/processor` and kernel
     operations (`op::rope`, `op::alibi_bias`) dictate how those coordinates are applied mathematically.

4. **routing and targeting tables:**
   - **`routing_ids[]`:** build a flat mapping array telling the `graph/processor` exactly which `memory/kv/block`
     or `memory/recurrent` slot to write each token's state into.
   - **`logit_targets[]`:** compute a list of indices marking which specific tokens require output logits
     (typically only the final token of a user's prompt). this prevents the kernel from wasting compute generating
     logits for intermediate prefill tokens.
