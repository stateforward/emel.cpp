---
title: kernel/vulkan architecture design
status: draft
---

# kernel/vulkan architecture design

this document defines kernel/vulkan. it executes typed kernel op events on
Vulkan-capable GPUs.

## role
- execute `op::*` events via Vulkan compute shaders.
- available on platforms with Vulkan support (Linux, Windows, Android).

## events (draft)
- `event::bind` inputs: gpu execution policy and device context.
- `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
- outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` handles incoming `op::*` events.
- unexpected non-op events route to `unexpected`.

## responsibilities
- map opcodes to Vulkan compute pipeline objects.
- record command buffers for each op event.
- manage Vulkan buffer bindings and descriptor sets per op event.
- submit and fence-wait for completion before dispatch returns.
