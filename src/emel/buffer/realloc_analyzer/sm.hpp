#pragma once

#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::realloc_analyzer {

/**
 * Buffer realloc analyzer orchestration model.
 *
 * Parity reference:
 * - `ggml_gallocr_node_needs_realloc(...)`
 * - `ggml_gallocr_needs_realloc(...)`
 *
 * Runtime invariants:
 * - Inputs are accepted only through `event::analyze`.
 * - Phase outcomes route through explicit `_done` / `_error` events only.
 * - Side effects are limited to writing output pointers from actions.
 * - No cross-machine mutation; analysis is pure over payload + snapshot.
 *
 * State purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates graph/snapshot payload contracts.
 * - `evaluating`: evaluates whether realloc is required.
 * - `publishing`: publishes `needs_realloc` boundary output.
 * - `done`: successful terminal before `events::analyze_done`.
 * - `failed`: failed terminal before `events::analyze_error`.
 * - `resetting`: clears runtime analysis state.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct idle {};
    struct validating {};
    struct evaluating {};
    struct publishing {};
    struct done {};
    struct failed {};
    struct resetting {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> / action::begin_analyze = sml::state<validating>,

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<evaluating>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<failed>,

      sml::state<evaluating> + sml::event<event::evaluate> / action::run_evaluate =
          sml::state<evaluating>,
      sml::state<evaluating> + sml::event<events::evaluate_done> = sml::state<publishing>,
      sml::state<evaluating> + sml::event<events::evaluate_error> = sml::state<failed>,

      sml::state<publishing> + sml::event<event::publish> / action::run_publish =
          sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<failed>,

      sml::state<done> + sml::event<events::analyze_done> / action::on_analyze_done =
          sml::state<idle>,
      sml::state<failed> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<evaluating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<publishing> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<done> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> + sml::event<events::reset_done> / action::on_reset_done =
          sml::state<idle>,
      sml::state<resetting> + sml::event<events::reset_error> / action::on_reset_error =
          sml::state<failed>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::analyze & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_analyze_error(phase_error);
    }
    if (!run_phase<event::evaluate, events::evaluate_done, events::evaluate_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_analyze_error(phase_error);  // GCOVR_EXCL_LINE
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_analyze_error(phase_error);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::analyze_done{
      .needs_realloc = context_.needs_realloc ? 1 : 0,
    });
  }

  bool process_event(const event::reset & ev) {
    int32_t phase_error = EMEL_OK;
    event::reset reset_ev = ev;
    reset_ev.error_out = &phase_error;
    if (!base_type::process_event(reset_ev)) return false;
    if (phase_error == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(events::reset_done{});
    }
    (void)base_type::process_event(events::reset_error{
      .err = phase_error,  // GCOVR_EXCL_LINE
    });
    return false;  // GCOVR_EXCL_LINE
  }

  bool needs_realloc() const noexcept { return context_.needs_realloc; }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {  // GCOVR_EXCL_BR_LINE
      error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return false;  // GCOVR_EXCL_LINE
    }
    if (error_out == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_analyze_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::analyze_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::buffer::realloc_analyzer
