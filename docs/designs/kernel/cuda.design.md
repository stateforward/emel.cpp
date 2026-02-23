---
title: kernel/cuda architecture design
status: draft
---

# kernel/cuda architecture design

this document defines kernel/cuda. it executes typed kernel op events on
NVIDIA GPUs via CUDA.

## role
- execute `op::*` events via CUDA kernels.
- available on platforms with NVIDIA CUDA support.

## events (draft)
- `event::bind` inputs: gpu execution policy and device context.
- `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
- outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` handles incoming `op::*` events.
- unexpected non-op events route to `unexpected`.

## responsibilities
- map opcodes to CUDA kernel launches.
- manage CUDA stream and device buffer bindings per op event.
- synchronize stream work for each op before dispatch returns.
