# Scaffolding plan (parity coverage)

This file captures the agreed machine/component scaffolding needed to reach full parity with
`tmp/llama.cpp` across decoder, tokenizer/encoder, generator, and parsing workflows. It
consolidates decisions from `docs/decoder.plan.md` and ongoing architecture discussions.

## Decoder
- `memory/coordinator/{recurrent,kv,hybrid}` with shared coordinator events and a common dispatch
  interface (see encoder for pattern).
- `decoder/sample_router` for backend sampling output mapping/copy/constraints.
- `batch/sanitizer` shared across decoder/encoder for batch validation + auto-generation.
- Decoder internal output handling in `decoder/output.hpp` for output capacity validation +
  ordering/reorder.
- `decoder/pooling.hpp` for pooling functions (constexpr pooler table; no separate SM).
- Shared `graph/scheduler` with decoder/encoder wrappers (`decoder/graph_scheduler`,
  `encoder/graph_scheduler`) replacing compute_* naming.

## Tokenizer + Encoder
- `tokenizer/pretokenizer`.
- `tokenizer/byte_fallback`.
- `tokenizer/specials.hpp` (helper module; no separate SM for now).
- Model encoder (inference) will live under `generator/encoder/{text,vision,audio}` since generator
  replaces `llama.cpp` context.

## Parsing + Templates
- `model/gguf/parser` for model file parsing/scaffolding.
- `gbnf/parser` for grammar parsing/scaffolding.
- `jinja/parser` for template parsing/scaffolding.
- `jinja/formatter` for template formatting/scaffolding.

## KV cache parity (same SM, expanded phases)
- Sequence operations remain in `kv/cache` (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`)
  with stream-aware scheduling.
- Extend `kv/cache` with stream-aware tracking and stream selection (`n_stream`, `seq_to_stream`,
  per-stream heads).
- Extend `kv/cache` with shift/defrag and sliding-window behaviors.

## Pending decisions
- Cross-attention bridge component naming/location (if needed).
- Placement for grammar constraint runtime (sampler vs decoder).
- Placement for prompt-template orchestration (generator vs text domain).
