# tokenizer preprocessor plan

this plan captures the agreed intent for a tokenizer preprocessor component and
its variants. it defines scope only; implementation will follow the existing
state machine conventions in `docs/rules/sml.rules.md`.

## summary
- the tokenizer needs a preprocessor stage that runs before encoding.
- the preprocessor is a model-aware preprocessor (not a token encoder).
- it normalizes and splits text into fragments for the tokenizer/encoder.
- it is not a single universal implementation; it varies by model/pre settings.

## responsibilities (preprocessor only)
- normalize and reshape input text without producing token ids.
- apply model-specific whitespace policies.
- isolate special tokens into fragments (without encoding them).
- produce a fragment list (raw spans + special token markers) for the tokenizer.

## non-responsibilities
- no token id generation.
- no encoder-specific merge logic (bpe/spm/wpm/ugm lives in encoder).
- no model selection beyond preprocessor variant binding.

## variants
- each variant is an SML machine under `tokenizer/preprocessor/<variant>`.
- selection is by `tokenizer_model_id`; variants may also branch on
  `tokenizer_pre_id` internally.
- planned variants:
  - `tokenizer/preprocessor/bpe` (regex-based splitting keyed by pre id).
  - `tokenizer/preprocessor/spm` (sentencepiece whitespace/escape behavior).
  - `tokenizer/preprocessor/ugm` (unigram whitespace policies).
  - `tokenizer/preprocessor/wpm` (wordpiece pre-splitting).
  - `tokenizer/preprocessor/rwkv` (minimal/identity normalization).
  - `tokenizer/preprocessor/plamo2` (minimal/identity normalization).
  - `tokenizer/preprocessor/fallback` (identity for unknown/none).

## interfaces (draft)
- each variant is a self-contained state machine with `actions`, `guards`,
  `events`, `context`, and `sm`.
- a shared preprocessor dispatch surface will be added under `tokenizer` to
  select and invoke the active variant based on model/vocab.

## open questions
- how much of current tokenizer fragment logic moves into preprocessor.
