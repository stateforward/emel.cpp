---
title: kernel/metal architecture design
status: draft
---

# kernel/metal architecture design

this document defines kernel/metal. it executes typed kernel op events on
Apple Metal GPUs.

## role
- execute `op::*` events via Metal compute shaders.
- available on macOS and iOS with Metal support.

## events (draft)
- `event::bind` inputs: gpu execution policy and device context.
- `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
- outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` handles incoming `op::*` events.
- unexpected non-op events route to `unexpected`.

## responsibilities
- map opcodes to Metal compute pipeline states.
- encode command buffers for each op event.
- manage Metal buffer bindings per op event.
- commit and wait for command buffer completion before dispatch returns.
