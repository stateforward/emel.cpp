---
title: text/encoders architecture design
status: rolling
---

# text/encoders architecture design

this document defines the text-domain encoder actor cluster that maps preprocessed text fragments to
token ids. this is distinct from model encoders (vision/audio/etc) and lives under the text domain
for clarity.

## role
- text/encoders is the algorithmic encoder cluster used by the tokenizer codec.
- it encodes raw text fragments (or preprocessed fragments) into token ids.
- it is model/vocab-aware but does not perform special-token parsing.

## scope
- text-domain tokenization only (BPE/SPM/WPM/UGM/RWKV/PLAMO2/fallback).
- excludes model encoders used for multimodal inputs.

## public interface (current contract)
- `event::encode`:
  - inputs: `vocab`, `text`, `preprocessed`, `token_ids`, `token_capacity`,
    `token_count_out`, `error_out`, optional sync callbacks.
  - outputs: `token_count_out` and `error_out`.

callbacks (`dispatch_done`/`dispatch_error`) are invoked synchronously and are not stored.

## composition
- per-algorithm SMs:
  - `bpe`, `spm`, `wpm`, `ugm`, `rwkv`, `plamo2`, `fallback`.
- each encoder SM uses the same single-step encode flow:
  `initialized` -> `encoding` -> `encode_decision` -> (`done` | `errored`),
  with explicit unexpected-event handling.
- `text/encoders::any` (sm_any) selects the active encoder kind and dispatches `event::encode`.

## invariants
- no allocations during dispatch.
- no special-token parsing; caller handles special-token policy.
- bounded work per encode request.

## error mapping
- invalid requests or capacity errors -> `EMEL_ERR_INVALID_ARGUMENT`.
- kernel/data errors propagate via `error_out`.

## status
- implemented under `src/emel/text/encoders/*` with namespace `emel::text::encoders`.

## open questions
- should `text/encoder` expose a unified `sm` alias or require `any` everywhere?
- how should byte-fallback support be represented (encoder vs tokenizer helper)?
