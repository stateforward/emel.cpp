# scaffolding plan (parity coverage)

this file captures the agreed machine/component scaffolding needed to reach full parity with
reference implementations across decoder, tokenizer/encoder, generator, and parsing workflows. it
consolidates decisions from `docs/decoder.plan.md` and ongoing architecture discussions.

## decoder
- `memory/coordinator/{recurrent,kv,hybrid}` with shared coordinator events and a common dispatch
  interface (see encoder for pattern).
- `decoder/sample_router` for backend sampling output mapping/copy/constraints.
- `batch/sanitizer` shared across decoder/encoder for batch validation + auto-generation.
- decoder internal output handling in `decoder/output.hpp` for output capacity validation +
  ordering/reorder.
- `decoder/pooling.hpp` for pooling functions (constexpr pooler table; no separate SM).
- shared `graph/scheduler` with decoder/encoder wrappers (`decoder/graph_scheduler`,
  `encoder/graph_scheduler`) replacing compute_* naming.

## tokenizer + encoder
- `tokenizer/preprocessor`.
- `tokenizer/byte_fallback`.
- `tokenizer/specials.hpp` (helper module; no separate SM for now).
- model encoder (inference) will live under `generator/encoder/{text,vision,audio}` since generator
  replaces `llama.cpp` context.

## parsing + templates
- `model/gguf/parser` for model file parsing/scaffolding.
- `gbnf/parser` for grammar parsing/scaffolding.
- `jinja/parser` for template parsing/scaffolding.
- `jinja/formatter` for template formatting/scaffolding.

## KV cache parity (same SM, expanded phases)
- sequence operations remain in `kv/cache` (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`)
  with stream-aware scheduling.
- extend `kv/cache` with stream-aware tracking and stream selection (`n_stream`, `seq_to_stream`,
  per-stream heads).
- extend `kv/cache` with shift/defrag and sliding-window behaviors.

## pending decisions
- cross-attention bridge component naming/location (if needed).
- placement for grammar constraint runtime (sampler vs decoder).
- placement for prompt-template orchestration (generator vs text domain).
