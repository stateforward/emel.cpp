# logits/sampler architecture design (draft)

this document defines the logits/sampler pipeline. it selects token ids from model logits.

## role
- logits/sampler is the output-selection stage: logits -> token ids.
- it is modality-agnostic and can be used by any generator.

## events (draft)
- `event::sample` inputs: logits pointer, vocab size, candidate buffers, sampler functions.
- `events::sample_done` outputs: selected token id.
- `events::sample_error` outputs: error_out.

## state model (draft)
- `initialized` -> `preparing` -> `prepare_decision`.
- `prepare_decision` -> (`sampling` | `done` | `errored`).
- `sampling` -> `sampling_decision` -> (`sampling` | `done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- validate inputs and candidate buffer capacities.
- run sampler functions in a bounded RTC loop.
- emit a selected token id or error.


## open questions
- should the sampler expose a unified `sample_step` for batched logits?
- how should per-sequence sampling parameters be carried across batched outputs?
