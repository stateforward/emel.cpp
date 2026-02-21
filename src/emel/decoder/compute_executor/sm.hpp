#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/actions.hpp"
#include "emel/decoder/compute_executor/events.hpp"
#include "emel/decoder/compute_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::compute_executor {

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
 * - `validating`/`validate_decision`: validate callbacks and ubatch inputs.
 * - `prepare_decision`: build or reuse compute graphs.
 * - `allocating_graph`/`alloc_decision`: allocate compute graph memory when required.
 * - `binding_inputs`/`bind_decision`: bind tensors for backend execution.
 * - `running_backend`/`backend_decision`: execute compute backend.
 * - `extracting_outputs`/`extract_decision`: read outputs for this ubatch.
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

}  // namespace emel::decoder::compute_executor
