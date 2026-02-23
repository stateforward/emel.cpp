---
title: kernel/any architecture design
status: draft
---

# kernel/any architecture design

this document defines kernel/any. it is the top-level `sm_any` dispatcher that routes
individual math operations to the active hardware backend.

## role
- receive compile-time typed opcode events (`op::*`) from `graph/processor` for individual nodes.
- delegate execution to the active hardware variant (e.g., `kernel::aarch64`, `kernel::metal`).
- provide an SML boundary for future async dispatch (`co_sm`) and graceful hardware fallbacks via `unexpected_event`.

## variants
the kernel domain is completely flat. `kernel::any` directly wraps the concrete hardware backends:
- `kernel::x86_64`
- `kernel::aarch64`
- `kernel::wasm`
- `kernel::cuda`
- `kernel::metal`
- `kernel::vulkan`

## events (draft)
- see `kernel/events.design.md` for full event catalog.
- `event::bind` inputs: kernel execution policy, hardware context.
- per-node: `op::*` events received from `graph/processor` and dispatched to the active variant via `make_dispatch_table`.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` handles incoming `op::*` events.
- unsupported operations on the active variant trigger `sml::unexpected_event`, routing to a CPU fallback.
- unexpected non-op events route to `unexpected`.

## responsibilities
- validate and bind hardware context for the active device.
- forward incoming `op::*` events directly to the active hardware variant.
- leverage SML's `unexpected_event` handling to catch unsupported opcodes on specific devices, allowing graceful fallbacks (e.g., routing an unsupported GPU op to the CPU fallback variant).
- propagate device execution errors back to the caller.
