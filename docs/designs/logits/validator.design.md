---
title: logits/validator architecture design
status: draft
---

# logits/validator architecture design

this document defines the logits/validator stage. it operates on a single sequence's logits,
validating and transforming raw model outputs into a structured candidate view for the sampler pipeline.

## role
- act as the candidate builder for a single sequence's logits row.
- validate candidate buffer capacities and raw logit pointers.
- normalize the view: pass through kernel-provided candidates (if a fused kernel like `op::top_k` was used)
  or gracefully fall back to building a full-vocab candidate list from raw logits.
- leave mathematical normalization (like Softmax) to the sampler chain.

## architecture shift: batch dispatch (future co_sm)
because a `batch::plan` step produces multiple rows of logits (one for each sequence requiring output),
the `generator` must process them efficiently.

initially, the validator is a standard synchronous `boost::sml` actor. the generator will loop over
the logits rows and process them sequentially.

in the future, by leveraging `emel::co_sm` (defined in `src/emel/sm.hpp`), the validator can be
dispatched asynchronously. the `generator` will be able to call `validator.process_event_async(...)`
for each sequence, and `co_await` their completion. the core SML state machine remains identical
in both synchronous and asynchronous contexts.

## events
- `event::validate`
  - inputs: logits row pointer, vocab size, optional kernel-provided candidates/counts,
    a scratch buffer for candidate generation, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: builds the normalized `candidate_view` in the provided buffers and invokes the
    appropriate callback before returning, avoiding context-reading race conditions.

## state model

```text
idle ──► validating ──► candidate_decision ──► (done | errored)
  ▲                                               │
  └───────────────────────────────────────────────┘
```

- `idle` — waiting for a logits row.
- `validating` — checking bounds and buffer capacities.
- `candidate_decision` — routing based on input:
  - if kernel candidates exist: simply wrap them in a `candidate_view`.
  - if raw logits only: zip the raw floats with token IDs (0 to vocab_size-1) into the scratch buffer.
- `done` — validation complete, transitions back to `idle` emitting `events::validate_done`.
- unexpected events route to `unexpected`.

## responsibilities
- **decouple generation from sampling:** ensure that whether the graph executed a highly fused `top_k`
  kernel or just dumped raw floats, the downstream sampler pipeline receives a consistent `candidate_view`.
- **no early math:** do not apply Softmax or modify scores. merely build the structural view. specific
  samplers in the chain will apply math in-place only if their algorithm requires it.
