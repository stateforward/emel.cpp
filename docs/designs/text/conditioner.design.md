---
title: text/conditioner architecture design
status: draft
---

# text/conditioner architecture design

this document defines the text/conditioner actor family. conditioners act as domain-specific input
processors for the modality-agnostic `generator`, formatting user intents into raw text strings and
orchestrating the `text/tokenizer` to produce raw token arrays.

## role
- act as an injected dependency for the `generator`.
- provide a configurable pipeline to format user inputs (raw text or structured chat messages) into
  model-specific prompt strings.
- own and orchestrate the `text/tokenizer` codec to translate those text strings into `token_id`s.
- yield raw, unsanitized token arrays back to the `generator`.

## architecture shift: dependency injection (`text/formatter`)
to provide both power-user flexibility and high-level ergonomics without complicating the SML state
machine, the `text/conditioner` relies on **explicit dependency injection**.

it does not format text itself. instead, it owns two injected dependencies:
1. **`text/formatter`:** a pure, stateless class that converts user input (e.g., raw strings or
   chat message arrays) into a single, contiguous string.
2. **`text/tokenizer`:** an SML actor that converts that string into `token_id`s.

when the user swaps a `formatter::raw` for a `formatter::chat`, the `conditioner`'s SML logic
remains completely unchanged. it simply orchestrates the handoff: Input -> Formatter -> Tokenizer -> Yield.

crucially, the conditioner does **not** perform batch slicing, sequence mapping, or memory validation.
it yields a raw array of `token_id`s to the `generator`, which uses its internal `token/batcher` to
handle the mathematical sanitation.

## events
- `event::bind`
  - inputs: `vocab`, injected `formatter` instance, and optional synchronous callbacks.
  - outputs: invokes callback upon successfully binding state ready to condition inputs.
- `event::prepare` (called by the user or the generator's step loop)
  - inputs: user prompt (domain-specific input payload) and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: passes the payload to the formatter, dispatches the resulting string to the tokenizer,
    populates the caller-provided buffer with a raw array of `token_id`s, and invokes the callback before returning.

## state model (general)

```text
uninitialized ──► binding ──► idle
                               │
idle ──► formatting ──► tokenizing ──► (idle | errored)
  ▲                                           │
  └───────────────────────────────────────────┘
```

- `uninitialized` — awaiting initial setup.
- `binding` — storing vocab references and binding the injected formatter.
- `idle` — waiting for a user prompt.
- `formatting` — synchronously calling the injected `text/formatter` to generate the prompt string.
- `tokenizing` — dispatching `event::tokenize` to the owned `text/tokenizer` to convert the string
  into `token_id`s.
- unexpected events route to `unexpected`.

## responsibilities
1. **flexible formatting:** provide a clean interface for users to choose their level of abstraction
   (raw text vs. auto-templated chat arrays) without muddying the core engine.
2. **tokenizer orchestration:** manage the synchronous invocation of the `text/tokenizer` actor,
   handling special-token parsing flags based on the formatting stage.
3. **raw yield:** provide a flat array of `token_id`s to the `generator`, delegating all sequence
   management and memory bounding to the generator's internal `token/batcher`.
