---
title: logits/sampler architecture design
status: draft
---

# logits/sampler architecture design

this document defines the logits/sampler pipeline. it takes a validated `candidate_view` for a single
sequence and passes it through a configurable chain of sampling algorithms to select the final token.

## role
- execute a strict, deterministic sequence of sampler algorithms (the "sampling chain").
- select a single token ID from the `candidate_view`.
- remain completely stateless across sequences and steps to allow pooling and reuse.

## architecture shift: sm_any and future co_sm
to efficiently handle batch decoding and dynamic sampler configurations without virtual function
overhead, this pipeline relies on `src/emel/sm.hpp`:

1. **`emel::sm_any` for dynamic chains:** the "sampling chain" (e.g., Repetition Penalty -> Temperature
   -> Top-K -> Top-P) is implemented as a `std::vector<emel::sm_any<sampler_kind, ...>>`. `sm_any` acts
   as a fast, type-erased wrapper for the individual SML sampler actors. the chain storage is prepared
   before dispatch; the pipeline actor simply loops through it, dispatching `event::apply` synchronously.
2. **future asynchronous execution (`emel::co_sm`):** initially, the pipeline is a standard
   synchronous `boost::sml` actor. in the future, it can be wrapped in `co_sm`, allowing the generator
   to dispatch `process_event_async` for each sequence concurrently and `co_await` completion.
   the underlying SML model requires no changes to support this transition.

## events
- `event::sample`
  - inputs: a validated `candidate_view`, a sampling chain policy (the vector of `sm_any`
    samplers), sequence-specific state (like the `rng` seed or previous tokens), and optional
    synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: sets the final selected `token_id` in the caller-provided destination and invokes the
    appropriate callback before returning, avoiding state machine context reads.

## state model

```text
idle ──► preparing_chain ──► applying_samplers ──► selecting_token ──► (done | errored)
  ▲                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

- `idle` — waiting for a `candidate_view`.
- `preparing_chain` — binding the `sm_any` chain and sequence-specific state (RNG, previous tokens) from
  the event payload.
- `applying_samplers` — a synchronous run-to-completion (RTC) loop that dispatches `event::apply` to
  each `sm_any` sampler in the chain sequentially. each sampler mutates the `candidate_view` in-place.
- `selecting_token` — after the chain finishes, pick the final token. if candidates remain, the final
  sampler (usually a multinomial or greedy sampler) will have moved the winning candidate to the front.
- `done` — selection complete, transitions back to `idle` emitting `events::sample_done`.

## responsibilities
- **stateless application:** the pipeline and its internal samplers must not store sequence data (like
  "previously generated tokens") in their SML context. all sequence-specific state is passed ephemerally
  in the `event::sample` payload. this allows a small pool of sampler pipelines to service thousands of
  sequences interchangeably.
- **in-place mutation:** samplers modify the `candidate_view` directly (e.g., by sorting it, truncating
  its count, or applying Softmax probabilities to the scores).
