---
title: text/detokenizer architecture design
status: draft
---

# text/detokenizer architecture design

this document defines the text/detokenizer actor. it is a pure, low-level codec stage that converts
token IDs directly into raw bytes or text pieces based on the model's vocabulary.

## role
- act as the pure translation layer: `token_id` -> `bytes`.
- remain completely stateless across sequences and unaware of generation logic, streaming, or stop
  conditions.
- honor model-specific vocabulary rules, including special-token handling and byte-fallback encoding.

## architecture shift: pure stateless codec
in dynamic systems, a detokenizer often holds internal state (like a pending byte buffer for partial
utf-8 characters) that ties it to a specific sequence.

in `emel`, to support efficient batch rendering, the `text/detokenizer` actor itself is stateless.
any required state (like a 4-byte pending utf-8 buffer) is owned by the parent `text/renderer` and
passed into the detokenizer via the `event::detokenize` payload. this allows a single, pooled
detokenizer actor to translate tokens for thousands of different sequences interchangeably.

## events
- `event::bind`
  - inputs: `vocab` reference and optional synchronous callbacks.
  - outputs: invokes callback upon successfully binding state ready for decode.
- `event::detokenize`
  - inputs: `token_id`, a reference to the renderer's pending byte buffer, `emit_special` flag,
    the output buffer capacity, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: writes utf-8 bytes to the output buffer, updates the pending byte buffer state, and
    invokes the appropriate callback before returning, completely avoiding context-reading race conditions.

## state model

```text
uninitialized ──► binding ──► idle
                               │
idle ──► decoding ──► decode_decision ──► (idle | errored)
  ▲                                          │
  └──────────────────────────────────────────┘
```

- `uninitialized` — awaiting initial setup.
- `binding` — validating the vocab reference.
- `idle` — waiting for a translation request.
- `decoding` — looking up the `token_id` in the vocabulary and applying byte-fallback rules.
- `decode_decision` — determining if the resulting bytes form a complete utf-8 sequence (combined
  with the pending buffer). if yes, write to output; if no, store in the pending buffer.
- unexpected events route to `unexpected`.

## decoding rules
1. **token-to-piece mapping:** uses vocab metadata to look up the text string for a given ID.
2. **special tokens:** if `emit_special` is false, tokens flagged as `special` (like `<|eot_id|>`)
   result in zero output bytes.
3. **byte-fallback accumulation:** if the token represents a raw byte (e.g., `<0xE2>`), it is
   appended to the renderer-provided pending buffer. the detokenizer only flushes bytes to the
   output buffer when they form a valid, complete utf-8 sequence.

## invariants
- no allocations during dispatch.
- output is written only to caller-provided buffers.
- the actor holds no sequence-specific state between dispatches.
