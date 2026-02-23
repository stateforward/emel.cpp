---
title: text/renderer architecture design
status: draft
---

# text/renderer architecture design

this document defines the text/renderer actor. it acts as a domain-specific output consumer for the
modality-agnostic `generator`. it receives raw token IDs, translates them to text, and handles
string-based stopping criteria and streaming formats.

## role
- act as an injected dependency (or external consumer) for the `generator`.
- receive raw `token_id` streams from the generator's sampling phase.
- own the `text/detokenizer` codec to translate tokens into utf-8 bytes.
- handle text-domain complexities: utf-8 boundary buffering, whitespace stripping, and string-based
  stop sequence matching.

## architecture shift: omnimodal decoupling
to support omnimodal generation (text, audio, vision), the core `generator` must remain completely
ignorant of what a `token_id` represents. it cannot own text-specific formatting or stop sequences.

the `generator` is passed a `renderer` (e.g., a `text/renderer` or an `audio/renderer`). when the
generator samples a new token, it passes it to the renderer. if the renderer detects that the newly
emitted text matches a user-defined string stop sequence (e.g., `"\nUser:"`), the renderer signals
back to the generator to halt that sequence.

## events
- `event::bind`
  - inputs: `vocab`, renderer options (e.g., stop sequences, strip whitespace flags), and optional callbacks.
  - outputs: invokes callback upon successfully binding state ready to render.
- `event::render` (called by the generator post-sampling)
  - inputs: `token_id`, sequence ID, output buffers (for streaming to the user), and optional callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: populates caller-provided buffers with translated utf-8 bytes (if any) and a `sequence_status` flag,
    invoking the appropriate callback before returning to prevent context reading.
- `event::flush`
  - inputs: sequence ID, output buffers, and optional callbacks.
  - outputs: forces the emission of any pending bytes into the output buffers and invokes the callback.

## state model

```text
uninitialized ──► binding ──► idle
                               │
idle ──► rendering ──► render_decision ──► (idle | errored)
  ▲                                           │
  └───────────────────────────────────────────┘
```

- `uninitialized` — awaiting initial setup.
- `binding` — storing vocab references and compiling stop sequence patterns.
- `idle` — waiting for a `token_id` from the generator.
- `rendering` — passing the token to the `text/detokenizer` and buffering the result.
- `render_decision` — evaluating the newly translated text against the stop sequence list.
- unexpected events route to `unexpected`.

## responsibilities
1. **utf-8 stream management:** modern models use byte-fallback tokens. a single `token_id` might be
   a partial utf-8 character (e.g., `0xE2`). the renderer maintains a tiny pending byte buffer per
   sequence and only flushes valid, complete utf-8 strings to the user's output buffer.
2. **stop sequence matching:** evaluate the rolling text window against user-provided stop strings.
   if a match occurs, truncate the output and return a `stop_sequence_matched` status to the generator.
3. **formatting:** handle optional policies like stripping leading spaces from the very first token
   generated.
