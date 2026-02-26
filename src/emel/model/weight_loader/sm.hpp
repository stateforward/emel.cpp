#pragma once

/*
design doc: docs/designs/model/weight_loader.design.md
 ---
 title: model/weight_loader architecture design
 status: draft
 ---
 
 # model/weight_loader architecture design
 
 this document defines the `model/weight_loader` actor. it acts as the bridge between parsed file metadata and the physical memory backing the model's weight tensors, supporting both zero-copy memory mapping (`mmap`) and streaming loads.
 
 ## role
 - act as a pure SML actor that consumes a list of parsed tensor offsets and binds them to physical storage.
 - abstract the complexities of OS-level memory mapping (e.g., `mmap` on Linux/macOS, `MapViewOfFile` on Windows) away from the mathematical execution pipeline.
 - support dynamic loading strategies, including chunked streaming over a network or split-file architectures, without exposing those details to the `generator`.
 - handle graceful fallback and cleanup if memory boundaries are exceeded.
 
 ## architecture shift: decoupled hardware binding
 in `llama.cpp`, the logic to open file descriptors, map memory, and construct tensors was tightly coupled to the parser.
 
 in `emel`, the `model/weight_loader` operates solely on the structural metadata provided by the `model/loader`. it does not know what a GGUF header is. it simply receives a directive: "Tensor X is a `Q4_K` array of size `[4096, 4096]` located at byte offset `0x10000`." The weight loader then orchestrates the OS calls to make that data accessible to the `kernel` backends.
 
 ## events
 - `event::load_weights`
   - inputs: a parsed structural map of the model (tensor offsets and sizes) and hardware policy preferences (e.g., `use_mmap = true`).
   - outputs: executes the loading strategy, verifying the physical data is accessible and aligned, and invokes a completion callback.
 
 ## state model
 
 ```text
 uninitialized ──► initialized
                       │
 initialized ──► selecting ──► strategy_decision
                                   ├──► (mmap) ──► initializing ──► loading_mmap ──► load_decision
                                   │
                                   └──► (stream) ──► loading_streamed ──► load_decision
                                                                                │
                  (done | errored) ◄── cleaning_up ◄── validating ◄─────────────┘
 ```
 
 - `selecting` — determines the optimal backend loading strategy based on the hardware context and user flags.
 - `initializing` — (mmap only) ensures the file size and alignment support zero-copy mapping.
 - `loading_mmap` — delegates to the OS to map the file into virtual memory.
 - `loading_streamed` — (alternative) allocates host/device memory and reads the file progressively (e.g., for split models or systems without `mmap` support).
 - `validating` — confirms that all required tensors are successfully backed by physical addresses or valid file descriptors.
 - `cleaning_up` — ensures temporary file handles or intermediate buffers are closed before returning control.
 - `done` — weights are bound and ready.
 
 ## responsibilities & constraints
 
 1. **strategy abstraction**:
    - the weight loader must seamlessly hide whether a tensor's data lives in RAM, VRAM, or on-disk via `mmap`. the downstream `graph` simply sees a `buffer` pointer.
    
 2. **zero-allocation hot path**:
    - this actor is invoked exactly once during model initialization. while it may allocate buffers during `loading_streamed`, it is never invoked during the `compute` phase, preserving `emel`'s strict inference rules.
 
 3. **deterministic cleanup**:
    - if a memory map fails (e.g., `ENOMEM`), the actor immediately routes to `errored`, executing cleanup actions to release partial mappings before signaling failure to the `model/loader`.
 
 ## effect boundary model
 
 the weight loader actor never performs I/O directly. all side effects — file opens, memory mapping, host/device allocation, byte transfers — happen outside the state machine.
 
 instead, the actor operates as a pure planner:
 
 1. **bind** — the orchestrator dispatches `event::bind_storage` with the tensor descriptor array produced by the parser. the actor binds references to these descriptors and transitions to `bound`.
 
 2. **plan** — the orchestrator dispatches `event::plan_load` with a loading policy (mmap, read-into-host, or read-into-device) and a fixed-capacity `effect_out` buffer. the actor walks the bound tensor descriptors and emits an ordered list of `effect_request` entries into the buffer. each entry describes a single I/O or allocation operation: file offset, byte count, destination pointer, and effect kind. the actor transitions to `awaiting_effects`.
 
 3. **execute** (external) — the orchestrator reads the effect request list and executes each operation outside SML dispatch. this is where actual `mmap` calls, `malloc`, device transfers, and file reads happen.
 
 4. **apply** — the orchestrator dispatches `event::apply_effect_results` with the completed handles and pointers. the actor validates that results match the planned effect list (count and kind), binds the resulting pointers into each tensor descriptor, and transitions to `ready`.
 
 this pattern keeps the actor pure and fully testable — you can verify the entire load plan without touching the filesystem. it also means the orchestrator has complete control over I/O scheduling, batching, and error handling.
 
 effect requests are emitted in a deterministic order: primary key is the tensor descriptor index, secondary key is the effect kind (map, alloc, copy). the actor never depends on OS allocation addresses for control flow.
 
 the actor's states under this model are:
 
 ```text
 unbound ──► bound ──► awaiting_effects ──► ready
               │            │                  │
               └────────────┴──────────────────┴──► errored
 ```
 
 - `unbound` — no tensor descriptors bound yet.
 - `bound` — tensor descriptors are bound; the actor is ready to plan.
 - `awaiting_effects` — the load plan has been emitted; the actor is waiting for the orchestrator to execute I/O and dispatch results.
 - `ready` — all weight pointers are bound and the model is ready for graph assembly.
 - `errored` — a validation failure, capacity overflow, or mismatched effect result.
 
 the actor does not retry implicitly. if a load fails, the orchestrator can dispatch `event::retry` explicitly, but backoff timing is owned entirely by the orchestrator.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_INVALID_ARGUMENT` — the loading strategy, tensor descriptors, or effect results are invalid, or apply results do not match the planned effect list.
 - `EMEL_ERR_CAPACITY` — the effect output buffer does not have sufficient capacity for the planned operations.
 - `EMEL_ERR_OOM` — out of memory during effect execution (reported by the orchestrator in the effect result, surfaced by the actor).
 - `EMEL_ERR_MODEL_INVALID` — the tensor descriptors are inconsistent with the model structure.
*/


#include "boost/sml.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::weight_loader {

struct unbound {};
struct bound {};
struct awaiting_effects {};
struct ready {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<unbound> + sml::event<event::bind_storage> [guard::valid_bind{}] /
        action::run_bind_storage = sml::state<bound>,
      sml::state<unbound> + sml::event<event::bind_storage> [guard::invalid_bind{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<bound> + sml::event<event::plan_load> [guard::valid_plan{}] /
        action::run_plan_load = sml::state<awaiting_effects>,
      sml::state<bound> + sml::event<event::plan_load> [guard::invalid_plan{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<awaiting_effects> + sml::event<event::apply_effect_results> [guard::valid_apply{}] /
        action::run_apply_effects = sml::state<ready>,
      sml::state<awaiting_effects> + sml::event<event::apply_effect_results> [guard::invalid_apply{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<ready> + sml::event<event::bind_storage> [guard::valid_bind{}] /
        action::run_bind_storage = sml::state<bound>,
      sml::state<errored> + sml::event<event::bind_storage> [guard::valid_bind{}] /
        action::run_bind_storage = sml::state<bound>,

      sml::state<unbound> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<bound> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<awaiting_effects> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<ready> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::bind_storage & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::bind_done{&ev});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::bind_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::plan_load & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::plan_done{&ev, context_.planned_effects});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::plan_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::apply_effect_results & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::apply_done{&ev});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::apply_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  using base_type::raw_sm;

  action::context context_{};
};

}  // namespace emel::model::weight_loader
