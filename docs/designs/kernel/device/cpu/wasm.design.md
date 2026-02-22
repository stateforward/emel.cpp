# kernel/device/cpu/wasm architecture design (draft)

this document defines kernel/device/cpu/wasm. it schedules graph instructions under WebAssembly.

## role
- schedule and compute graph instructions in a wasm runtime.
- select best ISA tier per-op (SIMD128 > scalar).

## events (draft)
- `event::schedule` inputs: bound `graph`, cpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `preparing` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- populate per-opcode function pointer table from compile-time feature flags.
- map opcodes to kernel functions (SIMD128 or scalar fallback).
- walk graph nodes sequentially, dispatching each to the selected kernel function.
- reuse scratch buffers across ops and steps.
- return only after all work is complete.
