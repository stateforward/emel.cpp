#pragma once

#include <cstdint>

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/decoder/ubatch_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::ubatch_executor {

struct initialized {};
struct validating {};
struct preparing_memory {};
struct preparing_kv {};
struct running_compute {};
struct extracting_outputs {};
struct rolling_back {};
struct done {};
struct errored {};

/**
 * Ubatch executor orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting ubatch execute intent.
 * - `validating`: validate dependencies before execution.
 * - `preparing_memory`/`preparing_kv`: prepare memory and kv cache.
 * - `running_compute`: run compute and kv apply for the ubatch.
 * - `extracting_outputs`: extract outputs after compute.
 * - `rolling_back`: attempt rollback after compute/output failure.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * Guard semantics:
 * - `valid_execute_request` is a pure predicate on the execute payload.
 * - `phase_*` guards observe errors set by entry actions.
 *
 * Action side effects:
 * - Entry actions run bounded submachine calls and update context fields.
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
      sml::state<validating> [guard::phase_ok] = sml::state<preparing_memory>,

      sml::state<preparing_memory> + sml::on_entry<sml::_> / action::run_prepare_memory_phase,
      sml::state<preparing_memory> [guard::phase_failed] = sml::state<errored>,
      sml::state<preparing_memory> [guard::phase_ok] = sml::state<preparing_kv>,

      sml::state<preparing_kv> + sml::on_entry<sml::_> / action::run_prepare_kv_phase,
      sml::state<preparing_kv> [guard::phase_failed] = sml::state<errored>,
      sml::state<preparing_kv> [guard::phase_ok] = sml::state<running_compute>,

      sml::state<running_compute> + sml::on_entry<sml::_> / action::run_compute_phase,
      sml::state<running_compute> [guard::phase_failed] = sml::state<rolling_back>,
      sml::state<running_compute> [guard::phase_ok] = sml::state<extracting_outputs>,

      sml::state<extracting_outputs> + sml::on_entry<sml::_> / action::run_extract_outputs_phase,
      sml::state<extracting_outputs> [guard::phase_failed] = sml::state<rolling_back>,
      sml::state<extracting_outputs> [guard::outputs_produced_invalid] /
          action::mark_missing_outputs = sml::state<rolling_back>,
      sml::state<extracting_outputs> [guard::phase_ok] = sml::state<done>,

      sml::state<rolling_back> + sml::on_entry<sml::_> / action::run_rollback_phase,
      sml::state<rolling_back> [guard::phase_failed] / action::capture_rollback_error =
          sml::state<errored>,
      sml::state<rolling_back> [guard::phase_ok] / action::capture_execution_error =
          sml::state<errored>,

      sml::state<done> + sml::on_entry<sml::_> / action::mark_done,
      sml::state<done> [guard::always] = sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> / action::ensure_last_error,
      sml::state<errored> [guard::always] = sml::state<initialized>,

      sml::state<initialized> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<validating> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_memory> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_kv> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<running_compute> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<extracting_outputs> + sml::event<event::execute> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<rolling_back> + sml::event<event::execute> / action::on_unexpected{} =
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
    if (ev.kv_tokens_out != nullptr) {
      *ev.kv_tokens_out = context_.kv_tokens;
    }
    if (ev.rollback_attempted_out != nullptr) {
      *ev.rollback_attempted_out = context_.rollback_attempted;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = context_.last_error;
    }
    return accepted && context_.last_error == EMEL_OK;
  }

  using base_type::process_event;

  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }
  int32_t kv_tokens() const noexcept { return context_.kv_tokens; }
  int32_t last_error() const noexcept { return context_.last_error; }
  bool rollback_attempted() const noexcept { return context_.rollback_attempted; }

 private:
  action::context context_{};
};

}  // namespace emel::decoder::ubatch_executor
