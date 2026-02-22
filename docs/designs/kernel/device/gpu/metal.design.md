# kernel/device/gpu/metal architecture design (draft)

this document defines kernel/device/gpu/metal. it schedules graph instructions on
Apple Metal GPUs.

## role
- schedule and compute graph instructions via Metal compute shaders.
- available on macOS and iOS with Metal support.

## events (draft)
- `event::schedule` inputs: bound `graph`, gpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- map opcodes to Metal compute pipeline states.
- encode command buffers from graph nodes.
- manage Metal buffer bindings for graph tensors.
- commit and wait for command buffer completion before returning done.
