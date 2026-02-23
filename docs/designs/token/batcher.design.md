---
title: token/batcher architecture design
status: draft
---

# token/batcher architecture design

this document defines token/batcher. it acts as the strict gateway between raw user input and the
generator's execution pipeline. it sanitizes, auto-populates, and validates token arrays, emitting a
canonical `token::batch` that the stateless `batch/planner` can trust completely.

## role
- act as a strict firewall: reject malformed token inputs (out-of-bounds IDs, invalid sequence
  positions) before they reach the graph or memory systems.
- auto-populate missing dimensions (e.g., sequence IDs, position IDs, logit masks) using "do what I
  mean" defaults based on the current `memory::any` state.
- emit a `token::batch`—a canonical, validated view of token IDs and metadata that serves as the
  single source of truth for the current generator cycle.

## architecture shift: the strict gateway
in dynamic systems (like `llama.cpp`), token validation is often intertwined with memory allocation
and planning (e.g., checking vocabulary bounds while simultaneously copying arrays into a new struct).

in `emel`, because the `batch/planner` is purely stateless and mathematical, the `token/batcher` must
guarantee that the `token::batch` it produces is logically sound. if the batcher succeeds, the
planner is guaranteed to be able to slice it without encountering semantic errors (like a sequence
jumping backwards in time).

## events
- `event::batch`
  - inputs: raw token IDs + token count, optional sequence IDs, optional position IDs,
    optional logit targets (output masks), and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: sanitizes the input, populates the canonical `token::batch` structures in the caller-provided
    buffers, and invokes the appropriate callback before returning, completely avoiding context reads.

## state model

```text
uninitialized ──► idle
                    │
idle ──► populating ──► validating ──► (done | errored)
  ▲                                       │
  └───────────────────────────────────────┘
```

- `uninitialized` — awaiting initial setup.
- `idle` — waiting for a raw token array.
- `populating` — filling in missing metadata (e.g., if positions are omitted, query `memory::any`
  for the current sequence length and auto-increment).
- `validating` — strictly enforcing logical continuity and vocabulary bounds.
- `done` — validation passed, transitions back to `idle` emitting `events::batch_done`.
- unexpected events route to `unexpected`.

## responsibilities

1. **auto-population (the "do what I mean" phase):**
   - **sequence IDs:** if omitted, default all tokens to a primary sequence (e.g., sequence `0`).
   - **positions:** if `position_ids` are omitted, query the `memory::any` interface to find the
     current length of the requested sequence(s), and auto-increment from there.
   - **logit masks:** if the user doesn't specify which tokens need output logits, default to `false`
     for all tokens *except* the very last token in the sequence (saving massive kernel compute).

2. **strict validation (the firewall):**
   - **vocabulary bounds:** reject any `token_id` that is `< 0` or `>= vocab.size()`.
   - **sequence continuity:** ensure the `position_ids` for any given sequence are strictly
     monotonically increasing. (e.g., reject `pos: [4, 5, 2, 3]` for sequence A).
   - **coupling constraints:** if two sequences share a token (e.g., during prompt sharing or
     prefix caching), ensure they agree on the position of that token. if they have diverged, reject.

3. **canonical emission:**
   - the resulting `token::batch` is not a copy of the data, but a structured, validated view
     (often just pointers to the user's arrays, plus the auto-populated metadata arrays).
   - this `token::batch` is handed directly to the `batch/planner` for zero-copy slicing.
