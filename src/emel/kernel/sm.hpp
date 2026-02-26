#pragma once

/*
design doc: docs/designs/kernel/kernel.design.md
 ---
 title: kernel architecture design
 status: draft
 ---
 
 # kernel architecture design
 
 this document defines the kernel domain. the kernel provides highly-optimized hardware
 abstraction layers wrapped in pure `boost::sml` state machines for secure, declarative execution.
 
 ## role
 - provide a flat, type-safe backend boundary (`kernel::any`) for mathematical operations on tensors.
 - receive individual `op::*` events from the `graph/processor` execution loop.
 - map high-level ops to low-level hardware intrinsics (AVX, NEON) or device backends (CUDA, Metal) via zero-overhead SML transition tables.
 - handle graceful hardware fallbacks (e.g., GPU to CPU) natively via `sml::unexpected_event`.
 
 ## architecture: flat variant boundary
 the kernel domain is structured around a single `kernel::any` interface that directly maps to
 concrete hardware variants. there are no intermediary `device`, `cpu`, or `gpu` wrappers.
 
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

 ## dispatch model
 `kernel/dispatch` is documentation-only and not a standalone actor.

 runtime dispatch behavior:
- hardware capability detection happens during kernel boundary construction.
 - `kernel::any` selects and binds a concrete variant (`x86_64`, `aarch64`, `wasm`, `cuda`, `metal`, `vulkan`).
 - graph execution dispatches typed `op::*` events directly to `kernel::any`.
 - concrete variant transition tables provide compile-time dispatch with minimal runtime overhead.

 fallback behavior:
 - unsupported operations in a variant surface as `sml::unexpected_event`.
 - `kernel::any` may route to a deterministic fallback chain (for example device to cpu) and must terminate with `EMEL_ERR_UNSUPPORTED_OP` at the final tier.
 - fallback routing is strictly acyclic.
 
 ## future-proofing (`co_sm`)
 by maintaining the kernel as an explicit boundary, operations can be upgraded to asynchronous
 dispatch in the future.
 the `kernel::any` boundary can be implemented with a `co_sm`, allowing the `graph/processor` to
 `co_await` GPU kernel launch while the host performs other useful work.

 ## ops model
 `kernel/ops` is documentation-only and not a standalone actor.

 operation execution contract:
 - every kernel op receives an immutable payload describing source tensors, destination tensor, and
   op-specific parameters.
 - ops write only to the destination tensor region described by `dst`.
 - ops do not allocate, block, or perform I/O during dispatch.
 - payloads are consumed synchronously and not retained after completion.

 determinism:
 - given identical backend state and identical input/event bytes, op results are identical.
 - backends that cannot guarantee deterministic numerics expose a deterministic bind setting.
 - deterministic bind requests must fail if a deterministic implementation is unavailable.

 unsupported operations:
 - backends do not silently drop unsupported ops.
 - unsupported ops surface as `EMEL_ERR_UNSUPPORTED` or as `sml::unexpected_event<op::...>` based
   on transition coverage.
 - fallback decisions remain in `kernel::any` policy and are deterministic/acyclic.

 safety invariants:
 - backends bounds-check event dimensions and derived indexing.
 - null data pointers and invalid regions fail deterministically.
 - integer overflow in index arithmetic fails with `EMEL_ERR_INVALID_ARGUMENT`.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel {

struct idle {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::scaffold> = sml::state<idle>,
      sml::state<idle> + sml::unexpected_event<sml::_> = sml::state<idle>
    );
  }
};

using sm = emel::sm<model>;

}  // namespace emel::kernel
