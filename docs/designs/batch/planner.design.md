# batch/planner architecture design (draft)

this document defines batch/planner. it converts a sanitized token batch into a `batch::plan`
that describes step scheduling and mapping tables.

## role
- convert `token::batch` into a `batch::plan`.
- define step boundaries and token ordering for execution.

## events (draft)
- `event::plan` inputs: `token::batch` (sanitized tokens + metadata), planning policy (step size,
  split mode).
- `events::plan_done` outputs: `batch::plan` (step schedule + mapping tables), status.
- `events::plan_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `planning` -> `plan_decision` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- compute step sizes and offsets.
- build token index lists for plan steps.
- compute expected output counts per step.
