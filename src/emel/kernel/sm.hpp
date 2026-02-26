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

#include "emel/emel.h"
#include "emel/kernel/actions.hpp"
#include "emel/kernel/context.hpp"
#include "emel/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/guards.hpp"

namespace emel::kernel {

struct ready {};
struct primary_dispatch {};
struct primary_decision {};
struct secondary_dispatch {};
struct secondary_decision {};
struct tertiary_dispatch {};
struct tertiary_decision {};
struct dispatch_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<primary_dispatch> <= *sml::state<ready> + sml::event<event::dispatch_scaffold>
                 [ guard::valid_dispatch{} ]
                 / action::begin_dispatch

      //------------------------------------------------------------------------------//
      // Primary backend phase.
      , sml::state<primary_decision> <= sml::state<primary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_primary

      , sml::state<ready> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_done{} ]
                 / action::dispatch_done

      , sml::state<secondary_dispatch> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_unsupported{} ]

      , sml::state<dispatch_decision> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_failed{} ]

      //------------------------------------------------------------------------------//
      // Secondary backend phase.
      , sml::state<secondary_decision> <= sml::state<secondary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_secondary

      , sml::state<ready> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_done{} ]
                 / action::dispatch_done

      , sml::state<tertiary_dispatch> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_unsupported{} ]

      , sml::state<dispatch_decision> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_failed{} ]

      //------------------------------------------------------------------------------//
      // Tertiary backend phase.
      , sml::state<tertiary_decision> <= sml::state<tertiary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_tertiary

      , sml::state<ready> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_done{} ]
                 / action::dispatch_done

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_unsupported{} ]
                 / action::mark_unsupported

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_failed{} ]

      //------------------------------------------------------------------------------//
      // Finalization.
      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<primary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<primary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<secondary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<secondary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<tertiary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::scaffold & ev) {
    event::scaffold_ctx ctx{};
    event::dispatch_scaffold evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == EMEL_OK;
  }
};

using Kernel = sm;

}  // namespace emel::kernel
