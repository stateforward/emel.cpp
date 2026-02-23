---
title: text/formatter architecture design
status: draft
---

# text/formatter architecture design

this document defines the text/formatter component. it is a pure, stateless dependency injected into
the `text/conditioner`. it handles the domain-specific logic of converting structured user intents
into contiguous prompt strings.

## role
- act as a pure, stateless dependency (not an SML actor) injected into the `text/conditioner`.
- translate user input variants (e.g., raw strings, arrays of chat messages) into a single,
  formatted `std::string` that the tokenizer can process.
- isolate messy, string-manipulation logic (like applying ChatML or Jinja templates) away from the
  core state machines.

## architecture: explicit dependency injection
by pulling formatting out of the `text/conditioner` SML actor and into an injected dependency, we
allow extreme flexibility without modifying core engine states. users can inject different formatters
based on their needs:

1. **`text/formatter::raw`:**
   - the pass-through formatter.
   - takes a `std::string` and returns it unmodified.
   - used for base models or when the user wants to apply prompt templates manually.
2. **`text/formatter::chat`:**
   - the ergonomic formatter.
   - takes a structured `std::vector<message>` (where a message has a `role` and `content`).
   - reads the model's chat template metadata (e.g., from GGUF) and constructs the exact string
     with appropriate special tokens (like `<|start_header_id|>` or `[INST]`).
3. **custom formatters:**
   - users can easily implement the formatter interface to support specific JSON schemas, tool-use
     framing, or novel prompt engineering techniques without touching the engine's SML actors.

## interface contract
the formatter provides a single, synchronous function call. it does not maintain state across calls.

- `format(input, options) -> std::string`
  - `input`: a variant or standardized payload containing the user's intent.
  - `options`: optional flags (e.g., whether to append the generation prompt like `\nAssistant:`).
  - returns: the fully constructed string ready for tokenization.
