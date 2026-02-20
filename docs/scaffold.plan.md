# Decoder scaffolding plan (consolidated)

This file captures the agreed machine/component scaffolding needed to reach full decoder parity,
based on `docs/decoder.plan.md`.

## Agreed scaffolding
- `memory/coordinator/{recurrent,kv,hybrid}` with shared coordinator events and a common dispatch
  interface (see encoder for pattern).
- `decoder/sample_router` for backend sampling output mapping/copy/constraints.
- `batch/sanitizer` shared across decoder/encoder for batch validation + auto-generation.
- Decoder internal output handling in `decoder/output.hpp` for output capacity validation +
  ordering/reorder.
- `decoder/pooling.hpp` for pooling functions (constexpr pooler table; no separate SM).
- Shared `graph/scheduler` with decoder/encoder wrappers (`decoder/graph_scheduler`,
  `encoder/graph_scheduler`) replacing compute_* naming.
- Tokenizer stages under `tokenizer/`:
  - `tokenizer/pretokenizer`
  - `tokenizer/byte_fallback`
  - `tokenizer/specials.hpp` (helper module; no separate SM for now)
- Model encoder (inference) will live under `generator/encoder` since generator replaces
  `llama.cpp` context, with subdomains:
  - `generator/encoder/text`
  - `generator/encoder/vision`
  - `generator/encoder/audio`
- `model/gguf/parser` for model file parsing/scaffolding.
- `gbnf/parser` for grammar parsing/scaffolding.
- `jinja/parser` for template parsing/scaffolding.
- `jinja/formatter` for template formatting/scaffolding.

## Pending naming decisions
- Cross-attention bridge component (if needed).
