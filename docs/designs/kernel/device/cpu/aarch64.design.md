# kernel/device/cpu/aarch64 architecture design (draft)

this document defines kernel/device/cpu/aarch64. it schedules graph instructions on ARM 64-bit cpus.

## role
- schedule and compute graph instructions on aarch64 hosts.
- select best ISA tier per-op at construction (AMX > NEON > scalar).

## events (draft)
- `event::schedule` inputs: bound `graph`, cpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `preparing` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- populate per-opcode function pointer table from runtime feature detection.
- map opcodes to kernel functions (AMX, NEON, or scalar fallback).
- walk graph nodes sequentially, dispatching each to the selected kernel function.
- reuse scratch buffers across ops and steps.
- return only after all cpu work is complete.
