# logits/validator architecture design (draft)

this document defines the logits/validator stage. it validates and normalizes logits and
candidate buffers before sampling.

## role
- logits/validator validates logits/candidate buffers and provides a normalized candidate view.
- it mirrors the llama.cpp fallback behavior (full-vocab candidates when none are provided).

## events (draft)
- `event::validate` inputs: logits pointer + vocab size, optional sampled candidates and counts
  (kernel-provided), candidate id/score buffers + capacity.
- `events::validate_done` outputs: normalized candidate view (ids + scores + count).
- `events::validate_error` outputs: error_out (invalid argument, kernel error).

## responsibilities
- ensure logits pointer is valid and vocab size > 0.
- validate candidate buffer capacity against required count.
- select candidate source:
  - kernel-provided candidates if present and counts > 0,
  - otherwise fall back to full-vocab candidates built from logits.
- ensure candidate count > 0.


## open questions
- should validator accept kernel-provided probabilities or logits subsets?
- should it normalize scores (e.g., softmax) or leave that to samplers?
