# text/tokenizer architecture design (rolling)

this document captures the tokenizer actor as implemented today. it reflects the current
orchestration model and data contracts; any structural changes still require explicit approval.

## role
- text/tokenizer is a codec-stage actor: text -> token ids.
- tokenizer composes a preprocessor + `text/encoders::any` pair selected from model metadata.

## public interface
- `event::bind` (vocab binding)
  - inputs: `vocab`, `error_out`, optional sync callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: `error_out` and callbacks; tokenizer transitions to `idle` on success.
- `event::tokenize` (encode text)
  - inputs: `vocab`, `text`, `add_special`, `parse_special`, `token_ids_out`,
    `token_capacity`, `token_count_out`, `error_out`, optional callbacks.
  - outputs: `token_count_out`, `error_out`, and done/error callbacks.

callbacks (`dispatch_done`/`dispatch_error`) are invoked synchronously before dispatch returns and
are not stored in context.

## composition and binding
- tokenizer owns:
  - `text/tokenizer::preprocessor::any` (variant preprocessor SM),
  - `text/encoders::any` (variant encoder SM).
- binding maps `vocab->tokenizer_model_id` to preprocessor + encoder kinds:
  `spm`, `bpe`, `wpm`, `ugm`, `rwkv`, `plamo2`, or `fallback`.
- binding is performed by the tokenizer SM via the `bind` event.
- intended DI extension: allow callers to inject preprocessor/encoder handles; if provided, the
  tokenizer uses them directly, otherwise it selects by model metadata. (current implementation
  always selects internally.)

## state model (current)
- bind flow:
  `uninitialized` -> `binding_preprocessor` -> `binding_preprocessor_decision`
  -> `binding_encoder` -> `binding_encoder_decision` -> `idle`.
- tokenization flow:
  `idle` -> `preprocessing` -> `preprocess_decision` -> `prefix_decision`
  -> `encoding_ready` -> (`encoding_token_fragment` | `encoding_raw_fragment`)
  -> `encoding_decision` -> `encoding_ready` (loop)
  -> `suffix_decision` -> `finalizing` -> `done`.
- errors route to `errored`; sequencing violations go to `unexpected`.

## preprocessing
- preprocessor emits a bounded fragment list (`k_max_fragments`) into context.
- fragments are either `token` (pre-resolved special token id) or `raw_text` spans.
- `parse_special` controls whether special tokens are extracted in preprocessing.

## encoding
- encoding iterates fragments in a bounded RTC loop.
- `token` fragments append directly to `token_ids_out`.
- `raw_text` fragments are encoded by the bound encoder via `text/encoders::any`.
- `preprocessed` flag is forwarded to the encoder.

## prefix/suffix rules (current)
- BOS: added when `add_special && vocab->add_bos`.
- SEP: added when `add_special && model_kind == wpm && vocab->add_sep`.
- EOS: added when `add_special && model_kind != wpm && vocab->add_eos`.
- prefix/suffix additions require capacity and valid ids.

## error mapping
- invalid request, capacity overflow, or unexpected events -> `EMEL_ERR_INVALID_ARGUMENT`.
- invalid vocab ids -> `EMEL_ERR_MODEL_INVALID`.
- kernel/preprocessor/encoder failures propagate their `error_out` values.

## invariants
- no allocation during dispatch; fragments live in fixed-size arrays.
- no self-dispatch; internal progress uses anonymous transitions.
- outputs are written only through request payloads (no persistent output buffers).

## tests (current)
- tokenizer orchestration tests for bind, prefix/suffix, capacity, and error paths.
- per-encoder and per-preprocessor unit tests.
