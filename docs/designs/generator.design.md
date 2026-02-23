---
title: generator architecture design
status: draft
---

# generator architecture design

this document defines the `generator`, the central orchestrator for a single-modality inference
session. it is the `emel` equivalent of `llama_context`, redesigned as a cohesive, high-level
state machine that abstracts away the complexity of hardware memory and graph execution.

## role
- act as the master orchestrator for a single-modality pipeline (e.g., text, audio).
- provide a simple, declarative public API for sequence lifecycle (allocate, branch, free).
- coordinate the entire generation loop: from raw input (via an injected conditioner) to mathematical
  execution (via the graph and memory), to raw output (via an injected renderer).
- remain strictly agnostic to *what* the modality is. it only knows it has an input processor that
  yields tokens and an output processor that consumes tokens.

## architecture shift: the single-modality pipeline
in `llama.cpp`, the `llama_context` is an entangled monolith. users must write hundreds of lines of
C boilerplate to tokenize a prompt, manually shift KV cache memory, allocate batches, and detokenize
outputs.

in `emel`, the `generator` provides a clean, single-modality pipeline. whether the user configures
it with a `text/chat/conditioner` or a `text/instruction/conditioner`, the generator's internal logic
remains identical: it asks the conditioner for raw token IDs, sanitizes and maps them via
`token/batcher` + `batch/planner`, pumps each planned step through `graph/graph`, samples the result,
and hands the new `token_id` to the `renderer` to get the final output string.

when `emel` expands to omnimodal execution, a higher-level `multimodal/generator` will simply route
inputs to multiple instances of these single-modality `generator`s running inside it.

## composition and explicit dependency injection (DI)
to maintain strict modality agnosticism and fast compile times, the core `generator` actor uses
**explicit dependency injection**. it does not know how to instantiate a text or audio pipeline, and
it does not include their headers.

- **strictly injected dependencies:**
  - `model` (weights, topology metadata, vocab).
  - `conditioner` interface (turns domain input into raw token ID arrays).
  - `renderer` interface (turns `token_id`s back into domain output).
- **owns (stateful hardware):**
  - `memory/hybrid` (the unified KV cache and recurrent state manager).
  - `graph/graph` (the DAG topology and compute executor).
- **orchestrates (stateless execution pipelines):**
  - `token/batcher` (sanitizes input).
  - `batch/planner` (slices batches to fit the allocator's watermark).
  - `logits/validator` & `logits/sampler` (selects the next token).

*note: "magic" auto-instantiation (e.g., reading model metadata to automatically build a
`text/conditioner` and `text/renderer` for a text model) happens in a higher-level factory or
wrapper class (like `emel::make_generator(...)`) completely outside this core SML actor.*

## events
- `event::reserve`
  - inputs: worst-case dimensions (max batch size, max context length), model metadata.
  - outputs: warms up the `graph/allocator`, sizes the `memory/hybrid` tensors, transitions to `idle`.
- `event::allocate_sequence` / `event::free_sequence`
  - inputs: sequence IDs.
  - outputs: multi-casts creation/destruction commands to the `memory/hybrid` actor.
- `event::branch_sequence`
  - inputs: parent sequence ID, new child sequence ID.
  - outputs: dispatches zero-copy reference linking to the `memory/hybrid` actor, enabling seamless
    tree-search or prefix sharing.
- `event::step` (the generation pump)
  - inputs: raw domain input (e.g., a text string), sequence mapping, termination policy.
  - outputs: drives the synchronous SML run-to-completion loop from `conditioner` ->
    `token/batcher` -> `batch/planner` -> `graph` -> `sampler` -> `renderer`. emits the final
    domain output (e.g., UTF-8 bytes).

## state model

```text
uninitialized ──► reserving ──► idle
                                 │
idle ──► conditioning ──► planning ──► executing ──► sampling ──► rendering ──► (done | errored)
  ▲                                                                                 │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

- `uninitialized` — awaiting memory bounds.
- `reserving` — dispatching `event::reserve` to the graph and memory actors to establish the static
  hardware watermark.
- `idle` — waiting for a lifecycle command or an `event::step`.
- `conditioning` — delegating to the injected `conditioner` to yield raw token ID arrays.
- `planning` — delegating to `token/batcher` + `batch/planner` to sanitize tokens and slice them
  into executable graph steps.
- `executing` — looping through the steps, dispatching `event::compute` to the `graph/graph`.
- `sampling` — for sequences requiring output, dispatching `event::sample` to the logits pipeline.
- `rendering` — handing the newly sampled `token_id`s to the injected `renderer` to get final output.
- `done` — step complete, looping back to `idle`.
- unexpected events route to `unexpected`.

## responsibilities
1. **the step pump:** abstract away the immense complexity of tensor math, batch slicing, and
   memory orchestration. the user simply calls `step(input)` and receives `output`.
2. **sequence lifecycle:** guarantee that sequence allocation, zero-copy branching, and freeing
   remain perfectly synchronized across the internal `memory/hybrid` and `graph/graph` state machines.
3. **modality agnosticism:** strictly enforce the boundary between mathematical execution (tokens)
   and domain interpretation (text, audio).
