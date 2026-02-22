# kernel/device/gpu/vulkan architecture design (draft)

this document defines kernel/device/gpu/vulkan. it schedules graph instructions on
Vulkan-capable GPUs.

## role
- schedule and compute graph instructions via Vulkan compute shaders.
- available on platforms with Vulkan support (Linux, Windows, Android).

## events (draft)
- `event::schedule` inputs: bound `graph`, gpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- map opcodes to Vulkan compute pipeline objects.
- record command buffers from graph nodes.
- manage Vulkan buffer bindings and descriptor sets for graph tensors.
- submit and fence-wait for command buffer completion before returning done.
