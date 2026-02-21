#pragma once

#include <type_traits>

#include "emel/sm.hpp"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/events.hpp"
#include "emel/tensor/lifetime_analyzer/guards.hpp"

namespace emel::tensor::lifetime_analyzer {

struct idle {};
struct validating {};
struct validate_decision {};
struct collecting_ranges {};
struct collect_decision {};
struct publishing {};
struct publish_decision {};
struct done {};
struct errored {};
struct reset_decision {};

/**
 * tensor lifetime analysis orchestration model.
 *
 * runtime invariants:
 * - inputs are accepted only through `event::analyze`.
 * - internal phases advance through anonymous transitions only.
 * - phase outcomes route through explicit error states.
 * - completion/error is explicit through terminal states.
 *
 * state purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates payload pointers/counts and output contracts.
 * - `collecting_ranges`: computes first/last-use ranges per tensor id.
 * - `publishing`: placeholder phase before output publication by caller.
 * - `done`: successful terminal.
 * - `errored`: failed terminal.
 * - `reset_decision`: clears runtime state and routes to `idle`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> / action::begin_analyze =
        sml::state<validating>,
      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok{}] = sml::state<collecting_ranges>,

      sml::state<collecting_ranges> / action::run_collect_ranges = sml::state<collect_decision>,
      sml::state<collect_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<collect_decision> [guard::phase_ok{}] = sml::state<publishing>,

      sml::state<publishing> / action::run_publish = sml::state<publish_decision>,
      sml::state<publish_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> = sml::state<idle>,
      sml::state<errored> = sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<validate_decision> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<collecting_ranges> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<collect_decision> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<publishing> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<publish_decision> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<done> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<errored> + sml::event<event::reset> / action::begin_reset =
        sml::state<reset_decision>,
      sml::state<reset_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<reset_decision> [guard::phase_ok{}] = sml::state<idle>,

      sml::state<idle> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<validating> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<validate_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<collecting_ranges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<collect_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<publishing> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<publish_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<reset_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::analyze & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    int32_t err = context_.phase_error;
    if (!accepted && err == EMEL_OK) {
      err = EMEL_ERR_BACKEND;
    }
    if (err == EMEL_OK) {
      if (ev.first_use_out != nullptr && ev.last_use_out != nullptr) {
        for (int32_t i = 0; i < context_.tensor_count; ++i) {
          ev.first_use_out[i] = context_.first_use[i];
          ev.last_use_out[i] = context_.last_use[i];
        }
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::reset_phase(context_);
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool process_event(const event::reset & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    int32_t err = context_.phase_error;
    if (!accepted && err == EMEL_OK) {
      err = EMEL_ERR_BACKEND;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::reset_phase(context_);
    return emel::detail::normalize_event_result(ev, accepted);
  }

  int32_t analyzed_tensor_count() const noexcept { return context_.tensor_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tensor::lifetime_analyzer
