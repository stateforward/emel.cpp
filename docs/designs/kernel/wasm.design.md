---
title: kernel/wasm architecture design
status: draft
---

# kernel/wasm architecture design

this document defines kernel/wasm. it executes typed kernel op events under WebAssembly.

## role
- execute `op::*` events in a wasm runtime.
- select best ISA tier per op (SIMD128 > scalar).

## events (draft)
- `event::bind` inputs: cpu execution policy and hardware context.
- `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
- outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` handles incoming `op::*` events.
- unexpected non-op events route to `unexpected`.

## responsibilities
- populate per-opcode dispatch table from compile-time feature flags.
- execute each received `op::*` event using SIMD128 or scalar fallback.
- reuse scratch buffers across op events.
- avoid graph traversal here; node walking stays in `graph/processor`.
