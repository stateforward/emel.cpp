---
title: text/tokenizer preprocessor design
status: draft
---

# text/tokenizer preprocessor design

this design captures the agreed intent for a tokenizer preprocessor component and
its variants. it defines scope only; implementation will follow the existing
state machine conventions in `docs/sml.rules.md`.

## summary
- the tokenizer needs a preprocessor stage that runs before encoding.
- the preprocessor is a model-aware preprocessor (not a token encoder).
- it normalizes and splits text into fragments for the tokenizer.
- it is not a single universal implementation; it varies by model/pre settings.

## responsibilities (preprocessor only)
- normalize and reshape input text without producing token ids.
- apply model-specific whitespace policies.
- isolate special tokens into fragments (without encoding them).
- produce a fragment list (raw spans + special token markers) for the tokenizer.

## variants
- each variant is an SML machine under `text/tokenizer/preprocessor/<variant>`.
- selection is by `tokenizer_model_id`; variants may also branch on
  `tokenizer_pre_id` internally.
- planned variants:
  - `text/tokenizer/preprocessor/bpe` (regex-based splitting keyed by pre id).
  - `text/tokenizer/preprocessor/spm` (sentencepiece whitespace/escape behavior).
  - `text/tokenizer/preprocessor/ugm` (unigram whitespace policies).
  - `text/tokenizer/preprocessor/wpm` (wordpiece pre-splitting).
  - `text/tokenizer/preprocessor/rwkv` (minimal/identity normalization).
  - `text/tokenizer/preprocessor/plamo2` (minimal/identity normalization).
  - `text/tokenizer/preprocessor/fallback` (identity for unknown/none).

## events (draft)
- `event::preprocess` inputs: input text, preprocessor policy, output fragment buffer +
  capacity.
- `events::preprocess_done` outputs: fragment list + count_out.
- `events::preprocess_error` outputs: error_out.
- each variant is a self-contained state machine with `actions`, `guards`, `events`,
  `context`, and `sm`.
- a shared preprocessor dispatch surface will be added under `tokenizer` to select and
  invoke the active variant based on model/vocab.

## open questions
- output buffer is caller-provided (`fragment *`, `capacity`, `count_out`).
- tokenizer integration deferred; preprocessor machines scaffolded only.
