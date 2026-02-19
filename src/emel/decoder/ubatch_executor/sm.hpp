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
struct validate_decision {};
struct prepare_memory_decision {};
struct prepare_kv_decision {};
struct compute_decision {};
struct extract_decision {};
struct rollback_decision {};
struct done {};
struct errored {};

/**
 * Ubatch executor orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting ubatch execute intent.
 * - `validating`/`validate_decision`: validate dependencies before execution.
 * - `preparing_memory`/`prepare_memory_decision`: prepare memory.
 * - `preparing_kv`/`prepare_kv_decision`: prepare kv cache.
 * - `running_compute`/`compute_decision`: run compute and kv apply for the ubatch.
 * - `extracting_outputs`/`extract_decision`: extract outputs after compute.
 * - `rolling_back`/`rollback_decision`: attempt rollback after compute/output failure.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * Guard semantics:
 * - `valid_execute_request` is a pure predicate on the execute payload.
 * - `phase_*` guards observe errors set by actions.
 * - `outputs_produced_invalid` enforces the expected per-ubatch output count.
 *
 * Action side effects:
 * - Actions run bounded submachine calls and update context fields.
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
      sml::state<validate_decision> [guard::phase_ok] = sml::state<preparing_memory>,

      sml::state<preparing_memory> / action::run_prepare_memory =
          sml::state<prepare_memory_decision>,
      sml::state<prepare_memory_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<prepare_memory_decision> [guard::phase_ok] = sml::state<preparing_kv>,

      sml::state<preparing_kv> / action::run_prepare_kv = sml::state<prepare_kv_decision>,
      sml::state<prepare_kv_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<prepare_kv_decision> [guard::phase_ok] = sml::state<running_compute>,

      sml::state<running_compute> / action::run_compute = sml::state<compute_decision>,
      sml::state<compute_decision> [guard::phase_failed] = sml::state<rolling_back>,
      sml::state<compute_decision> [guard::phase_ok] = sml::state<extracting_outputs>,

      sml::state<extracting_outputs> / action::run_extract_outputs =
          sml::state<extract_decision>,
      sml::state<extract_decision> [guard::phase_failed] = sml::state<rolling_back>,
      sml::state<extract_decision> [guard::outputs_produced_invalid] /
          action::mark_missing_outputs = sml::state<rolling_back>,
      sml::state<extract_decision> [guard::phase_ok] = sml::state<done>,

      sml::state<rolling_back> / action::run_rollback = sml::state<rollback_decision>,
      sml::state<rollback_decision> [guard::phase_failed] / action::capture_rollback_error =
          sml::state<errored>,
      sml::state<rollback_decision> [guard::phase_ok] / action::capture_execution_error =
          sml::state<errored>,

      sml::state<done> / action::mark_done = sml::state<initialized>,

      sml::state<errored> / action::ensure_last_error = sml::state<initialized>,

      sml::state<validating> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<validate_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<preparing_memory> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<prepare_memory_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<preparing_kv> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<prepare_kv_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<running_compute> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<compute_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<extracting_outputs> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<extract_decision> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<rolling_back> + sml::event<event::execute> / action::on_unexpected =
          sml::state<errored>,
      sml::state<rollback_decision> + sml::event<event::execute> / action::on_unexpected =
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
