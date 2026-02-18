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
struct preparing_graph {};
struct allocating_graph {};
struct binding_inputs {};
struct running_backend {};
struct extracting_outputs {};
struct done {};
struct errored {};

/**
 * Compute executor orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting execute intent.
 * - `validating`: validate callbacks and ubatch inputs.
 * - `preparing_graph`/`allocating_graph`: build or reuse compute graphs.
 * - `binding_inputs`: bind tensors for backend execution.
 * - `running_backend`: execute compute backend.
 * - `extracting_outputs`: read outputs for this ubatch.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * Guard semantics:
 * - `valid_execute_request` is a pure predicate on the execute payload.
 * - `phase_*` guards observe errors set by entry actions.
 *
 * Action side effects:
 * - Entry actions run bounded compute steps and update context fields.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::execute> [guard::valid_execute_request{}] /
          action::begin_execute = sml::state<validating>,
      sml::state<initialized> + sml::event<event::execute> [guard::invalid_execute_request{}] /
          action::reject_invalid_execute = sml::state<errored>,

      sml::state<validating> + sml::on_entry<sml::_> / action::run_validate_phase,
      sml::state<validating> [guard::phase_failed] = sml::state<errored>,
      sml::state<validating> [guard::phase_ok] = sml::state<preparing_graph>,

      sml::state<preparing_graph> + sml::on_entry<sml::_> / action::run_prepare_graph_phase,
      sml::state<preparing_graph> [guard::phase_failed] = sml::state<errored>,
      sml::state<preparing_graph> [guard::graph_reused] = sml::state<binding_inputs>,
      sml::state<preparing_graph> [guard::graph_needs_allocation] =
          sml::state<allocating_graph>,

      sml::state<allocating_graph> + sml::on_entry<sml::_> / action::run_alloc_graph_phase,
      sml::state<allocating_graph> [guard::phase_failed] = sml::state<errored>,
      sml::state<allocating_graph> [guard::phase_ok] = sml::state<binding_inputs>,

      sml::state<binding_inputs> + sml::on_entry<sml::_> / action::run_bind_inputs_phase,
      sml::state<binding_inputs> [guard::phase_failed] = sml::state<errored>,
      sml::state<binding_inputs> [guard::phase_ok] = sml::state<running_backend>,

      sml::state<running_backend> + sml::on_entry<sml::_> / action::run_backend_phase,
      sml::state<running_backend> [guard::phase_failed] = sml::state<errored>,
      sml::state<running_backend> [guard::phase_ok] = sml::state<extracting_outputs>,

      sml::state<extracting_outputs> + sml::on_entry<sml::_> / action::run_extract_outputs_phase,
      sml::state<extracting_outputs> [guard::phase_failed] = sml::state<errored>,
      sml::state<extracting_outputs> [guard::phase_ok] = sml::state<done>,

      sml::state<done> + sml::on_entry<sml::_> / action::mark_done,
      sml::state<done> [guard::always] = sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> / action::ensure_last_error,
      sml::state<errored> [guard::always] = sml::state<initialized>,

      sml::state<initialized> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<validating> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_graph> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<allocating_graph> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<binding_inputs> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<running_backend> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<extracting_outputs> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<done> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<errored> + sml::event<event::execute> / action::on_unexpected{} =
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
