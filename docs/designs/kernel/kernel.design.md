---
title: kernel architecture design
status: draft
---

# kernel architecture design

this document defines the kernel domain. the kernel provides highly-optimized hardware
abstraction layers wrapped in pure `boost::sml` state machines for secure, declarative execution.

## role
- provide a flat, type-safe actor boundary (`kernel::any`) for mathematical operations on tensors.
- receive individual `op::*` events from the `graph/processor` execution loop.
- map high-level ops to low-level hardware intrinsics (AVX, NEON) or device backends (CUDA, Metal) via zero-overhead SML transition tables.
- handle graceful hardware fallbacks (e.g., GPU to CPU) natively via `sml::unexpected_event`.

## architecture: the flat `sm_any` hierarchy
the kernel domain is structured as a single `kernel::any` wrapper that directly manages concrete hardware variants. there are no intermediary `device`, `cpu`, or `gpu` wrappers.

available variants:
- `kernel::x86_64` (x86_64 CPUs, managing scalar, AVX2, AVX512 internally)
- `kernel::aarch64` (ARM CPUs, managing scalar, NEON, AMX internally)
- `kernel::wasm` (WebAssembly SIMD)
- `kernel::cuda` (NVIDIA GPUs)
- `kernel::metal` (Apple GPUs)
- `kernel::vulkan` (Cross-platform GPUs)

## the hot path (zero-overhead dispatch)
the `graph/processor` acts as the execution loop, walking the DAG and synchronously dispatching `op::*` events to `kernel::any`.
because `boost::sml` transition tables are resolved at compile time into optimized jump tables or `switch` statements, the cost of dispatching an `op::*` event is practically zero (equivalent to a function pointer call).

this architecture provides absolute performance parity with traditional C-style function pointer dispatch (like `llama.cpp`), while keeping the entire compute execution secure within the Run-To-Completion actor model.

## future-proofing (`co_sm`)
by maintaining the kernel as an actor boundary, operations can easily be upgraded to asynchronous dispatches in the future.
the `kernel::any` actor can be swapped for a `co_sm`, allowing the `graph/processor` to `co_await` the launch of a GPU compute kernel, yielding the host thread to perform other tasks (like tokenizing inputs or handling network I/O) while the hardware executes the math.
