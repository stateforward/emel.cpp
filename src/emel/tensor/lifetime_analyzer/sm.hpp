#pragma once

#include "emel/sm.hpp"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/events.hpp"
#include "emel/tensor/lifetime_analyzer/guards.hpp"

namespace emel::tensor::lifetime_analyzer {

/**
 * Tensor lifetime analysis orchestration model.
 *
 * Runtime invariants:
 * - Inputs are accepted only through `event::analyze`.
 * - Phase outcomes route through explicit `_done` / `_error` events only.
 * - All state mutation and side effects (writing output arrays) occur in actions.
 * - Completion/error is explicit through `_done` / `_error` events.
 *
 * State purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates payload pointers/counts and output contracts.
 * - `collecting_ranges`: computes first/last-use ranges per tensor id.
 * - `publishing`: writes computed ranges to output buffers.
 * - `done`: successful terminal before `events::analyze_done`.
 * - `failed`: failed terminal before `events::analyze_error`.
 * - `resetting`: clears runtime state.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct idle {};
    struct validating {};
    struct collecting_ranges {};
    struct publishing {};
    struct done {};
    struct failed {};
    struct resetting {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> / action::begin_analyze =
          sml::state<validating>,

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> =
          sml::state<collecting_ranges>,
      sml::state<validating> + sml::event<events::validate_error> =
          sml::state<failed>,

      sml::state<collecting_ranges> + sml::event<event::collect_ranges> /
          action::run_collect_ranges = sml::state<collecting_ranges>,
      sml::state<collecting_ranges> + sml::event<events::collect_ranges_done> =
          sml::state<publishing>,
      sml::state<collecting_ranges> + sml::event<events::collect_ranges_error> =
          sml::state<failed>,

      sml::state<publishing> + sml::event<event::publish> / action::run_publish =
          sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> =
          sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> =
          sml::state<failed>,

      sml::state<done> + sml::event<events::analyze_done> / action::on_analyze_done =
          sml::state<idle>,
      sml::state<failed> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<collecting_ranges> + sml::event<event::reset> / action::begin_reset =
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
    if (!run_phase<event::collect_ranges, events::collect_ranges_done, events::collect_ranges_error>(
            phase_error)) {
      return finalize_analyze_error(phase_error);
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(phase_error)) {
      return finalize_analyze_error(phase_error);
    }
    return base_type::process_event(events::analyze_done{});
  }

  bool process_event(const event::reset & ev) {
    int32_t phase_error = EMEL_OK;
    event::reset reset_ev = ev;
    reset_ev.error_out = &phase_error;
    if (!base_type::process_event(reset_ev)) return false;
    if (phase_error == EMEL_OK) {
      return base_type::process_event(events::reset_done{});
    }
    (void)base_type::process_event(events::reset_error{
      .err = phase_error,
    });
    return false;
  }

  int32_t analyzed_tensor_count() const noexcept { return context_.tensor_count; }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {
      error_out = EMEL_ERR_BACKEND;
      return false;
    }
    if (error_out == EMEL_OK) {
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

}  // namespace emel::tensor::lifetime_analyzer
