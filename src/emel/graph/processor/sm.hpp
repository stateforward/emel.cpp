#pragma once

/*
design doc: docs/designs/graph/processor.design.md
 ---
 title: graph/processor architecture design
 status: draft
 ---
 
 # graph/processor architecture design
 
 this document defines graph/processor. it binds a prepared graph for one plan step and dispatches
 kernel execution.
 
 ## role
 - bind a prepared graph topology for one plan step, dispatch kernel execution, and extract outputs.
 - act as the runtime-to-compile-time bridge (the "Opcode Router") between the dynamic DAG and the
   strongly-typed `kernel::any` actor.
 - procedurally manage tensor lifecycles during the hot loop by decrementing reference counts in the
   graph's DOD arrays as nodes complete.
 
 ## architecture shift: the opcode router
 the `kernel` domain achieves zero-overhead execution by relying on compile-time typed events (e.g.,
 `kernel::op::add{...}`). however, the execution DAG is built dynamically at runtime (e.g., parsing a
 GGUF file), meaning a tensor node only holds a runtime enum representing its operation
 (e.g., `node->opcode == OP_ADD`).
 
 to bridge this gap, the `graph/processor` acts as the **Opcode Router**. during its execution walk,
 it reads the runtime `node->opcode` enum and uses `sml::utility::make_dispatch_table` (or a static
 `switch` statement) to construct the corresponding strongly-typed `op::*` event and dispatch it to
 `kernel::any`. this isolates the dynamic-to-static transition to a single, highly optimized location
 in the hot path.
 
 ## composition
 - owned by graph::sm.
 
 ## events (draft)
 - `event::process`
   - inputs: bound DAG topology (from graph::sm context), `batch::plan` (token batch pointers +
     step mapping), `memory::any` (prepared view), output buffers for logits or pooled embeddings,
     and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: produced row counts + row mapping in the provided buffers, invoking the appropriate
     callback before returning to prevent caller context reads.
 
 ## state model (draft)
 - `idle` -> `binding_inputs` -> `running` -> `extracting` -> `done`.
 - failures route to `errored`, unexpected events to `unexpected`.
 
 ## responsibilities
 - bind step inputs into the graph.
 - execute a blazing-fast procedural `for` loop over the DAG nodes in topological order.
 - use the Opcode Router to synchronously dispatch math operations as `op::*` events to `kernel::any`.
 - immediately decrement the `refs` count in the graph's DOD tensor arrays for a node's sources after
   its kernel operation completes.
 - extract outputs into provided buffers.
 
 ## encode vs execute mode
 
 the processor's behavior depends on the kernel backend type:
 
 - **CPU backends** execute operations inline and return host-ready results immediately. the
   processor sets `requires_barrier = false` after the loop completes. the graph can transition
   directly to `done`.
 - **GPU backends** encode commands into a command buffer (CUDA stream, Metal command buffer, Vulkan
   command buffer) but do not wait for completion. the processor sets `requires_barrier = true`.
   the actual wait happens at the barrier level, driven by the graph or the generator's
   orchestrator.
 
 in both modes, the processor's execution loop is identical — it walks nodes in topological order
 and dispatches `op::*` events to `kernel::any`. the difference is entirely in what the kernel
 does with each event (execute immediately vs encode for later).
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid bounds or opcode ids were encountered during execution.
 - `EMEL_ERR_INTERNAL` — a tensor lifecycle invariant was violated (e.g., ref count underflow).
 - `EMEL_ERR_CAPACITY` — capacity overflow when constructing typed events or output descriptors.
 
 kernel failure codes propagate unchanged through the processor.
*/


#include <cstdint>

#include "emel/graph/processor/actions.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/graph/processor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::graph::processor {

struct initialized {};
struct validating {};
struct allocating_graph {};
struct binding_inputs {};
struct running_backend {};
struct extracting_outputs {};
struct validate_decision {};
struct prepare_decision {};
struct alloc_decision {};
struct bind_decision {};
struct backend_decision {};
struct extract_decision {};
struct done {};
struct errored {};

/**
 * compute executor orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting execute intent.
 * - `validating`/`validate_decision`: validate callbacks and step inputs.
 * - `prepare_decision`: build or reuse compute graphs.
 * - `allocating_graph`/`alloc_decision`: allocate compute graph memory when required.
 * - `binding_inputs`/`bind_decision`: bind tensors for backend execution.
 * - `running_backend`/`backend_decision`: execute compute backend.
 * - `extracting_outputs`/`extract_decision`: read outputs for this step.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * guard semantics:
 * - `valid_execute_request` is a pure predicate on the execute payload.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
 * - actions run bounded compute steps and update context fields.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::execute> [guard::valid_execute_request{}] /
          action::begin_execute = sml::state<validating>,
      sml::state<initialized> + sml::event<event::execute> [guard::invalid_execute_request{}] /
          action::reject_invalid_execute = sml::state<errored>,

      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok] / action::run_prepare_graph =
          sml::state<prepare_decision>,

      sml::state<prepare_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<prepare_decision> [guard::graph_reused] = sml::state<binding_inputs>,
      sml::state<prepare_decision> [guard::graph_needs_allocation] =
          sml::state<allocating_graph>,

      sml::state<allocating_graph> / action::run_alloc_graph = sml::state<alloc_decision>,
      sml::state<alloc_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<alloc_decision> [guard::phase_ok] = sml::state<binding_inputs>,

      sml::state<binding_inputs> / action::run_bind_inputs = sml::state<bind_decision>,
      sml::state<bind_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<bind_decision> [guard::phase_ok] = sml::state<running_backend>,

      sml::state<running_backend> / action::run_backend = sml::state<backend_decision>,
      sml::state<backend_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<backend_decision> [guard::phase_ok] = sml::state<extracting_outputs>,

      sml::state<extracting_outputs> / action::run_extract_outputs = sml::state<extract_decision>,
      sml::state<extract_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<extract_decision> [guard::phase_ok] = sml::state<done>,

      sml::state<done> / action::mark_done = sml::state<initialized>,

      sml::state<errored> / action::ensure_last_error = sml::state<initialized>,

      sml::state<validating> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<validate_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<prepare_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<allocating_graph> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<alloc_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<binding_inputs> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<bind_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<running_backend> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<backend_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<extracting_outputs> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<extract_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<done> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<errored> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::execute & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (ev.outputs_produced_out != nullptr) {
      *ev.outputs_produced_out = context_.outputs_produced;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = context_.last_error;
    }
    return accepted && context_.last_error == EMEL_OK;
  }

  using base_type::process_event;

  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }
  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  action::context context_{};
};

}  // namespace emel::graph::processor
