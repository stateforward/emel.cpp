#pragma once

#include <type_traits>

#include "emel/sm.hpp"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/events.hpp"
#include "emel/tensor/lifetime_analyzer/guards.hpp"

namespace emel::tensor::lifetime_analyzer {

struct Idle {};
struct Validating {};
struct ValidateDecision {};
struct CollectingRanges {};
struct CollectDecision {};
struct Publishing {};
struct PublishDecision {};
struct Done {};
struct Errored {};
struct ResetDecision {};

/**
 * Tensor lifetime analysis orchestration model.
 *
 * Runtime invariants:
 * - Inputs are accepted only through `event::analyze`.
 * - Internal phases advance through anonymous transitions only.
 * - Phase outcomes route through explicit error states.
 * - Completion/error is explicit through terminal states.
 *
 * State purposes:
 * - `Idle`: accepts `event::analyze` and `event::reset`.
 * - `Validating`: validates payload pointers/counts and output contracts.
 * - `CollectingRanges`: computes first/last-use ranges per tensor id.
 * - `Publishing`: placeholder phase before output publication by caller.
 * - `Done`: successful terminal.
 * - `Errored`: failed terminal.
 * - `ResetDecision`: clears runtime state and routes to `Idle`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<Idle> + sml::event<event::analyze> / action::begin_analyze =
        sml::state<Validating>,
      sml::state<Validating> / action::run_validate = sml::state<ValidateDecision>,
      sml::state<ValidateDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<ValidateDecision> [guard::phase_ok{}] = sml::state<CollectingRanges>,

      sml::state<CollectingRanges> / action::run_collect_ranges = sml::state<CollectDecision>,
      sml::state<CollectDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<CollectDecision> [guard::phase_ok{}] = sml::state<Publishing>,

      sml::state<Publishing> / action::run_publish = sml::state<PublishDecision>,
      sml::state<PublishDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<PublishDecision> [guard::phase_ok{}] = sml::state<Done>,

      sml::state<Done> = sml::state<Idle>,
      sml::state<Errored> = sml::state<Idle>,

      sml::state<Idle> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<Validating> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<ValidateDecision> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<CollectingRanges> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<CollectDecision> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<Publishing> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<PublishDecision> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<Done> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<Errored> + sml::event<event::reset> / action::begin_reset =
        sml::state<ResetDecision>,
      sml::state<ResetDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<ResetDecision> [guard::phase_ok{}] = sml::state<Idle>,

      sml::state<Idle> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<Validating> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ValidateDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<CollectingRanges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<CollectDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<Publishing> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<PublishDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<Done> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<Errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ResetDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>
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
