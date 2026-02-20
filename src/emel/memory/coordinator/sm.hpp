#pragma once

#include <cstdint>
#include <type_traits>

#include "emel/emel.h"
#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::memory::coordinator {

struct initialized {};

struct validating_update {};
struct validating_batch {};
struct validating_full {};

struct preparing_update {};
struct prepare_update_decision {};

struct preparing_batch {};
struct prepare_batch_decision {};

struct preparing_full {};
struct prepare_full_decision {};

struct applying_update {};
struct apply_update_decision {};

struct publishing_update {};
struct publish_update_decision {};
struct publishing_batch {};
struct publish_batch_decision {};
struct publishing_full {};
struct publish_full_decision {};

struct done {};
struct errored {};

/**
 * Memory coordination orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting a prepare request.
 * - `validating_*`: request validation via guards.
 * - `preparing_*`: compute prepared status and update context.
 * - `apply_*`: apply pending updates for update requests.
 * - `publishing_*`: publish final status.
 * - `*_decision`: branch on `phase_error` and prepared status.
 * - `done`/`errored`: terminal outcomes that return to initialized.
 *
 * Guard semantics:
 * - `valid_*` guards are pure predicates over `(context)`.
 * - `phase_*` guards observe `context.phase_error`.
 * - `prepare_update_*` guards inspect `context.prepared_status`.
 *
 * Action side effects:
 * - Actions update context counters and `prepared_status`.
 * - Errors are recorded in `phase_error`/`last_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare_update> /
          action::begin_prepare_update = sml::state<validating_update>,
      sml::state<initialized> + sml::event<event::prepare_batch> /
          action::begin_prepare_batch = sml::state<validating_batch>,
      sml::state<initialized> + sml::event<event::prepare_full> /
          action::begin_prepare_full = sml::state<validating_full>,

      sml::state<validating_update> [guard::valid_update_context{}] =
          sml::state<preparing_update>,
      sml::state<validating_update> [guard::invalid_update_context{}] /
          action::set_invalid_argument = sml::state<errored>,

      sml::state<validating_batch> [guard::valid_batch_context{}] =
          sml::state<preparing_batch>,
      sml::state<validating_batch> [guard::invalid_batch_context{}] /
          action::set_invalid_argument = sml::state<errored>,

      sml::state<validating_full> [guard::valid_full_context{}] =
          sml::state<preparing_full>,
      sml::state<validating_full> [guard::invalid_full_context{}] /
          action::set_invalid_argument = sml::state<errored>,

      sml::state<preparing_update> / action::run_prepare_update_phase =
          sml::state<prepare_update_decision>,
      sml::state<prepare_update_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<prepare_update_decision> [guard::prepare_update_success{}] =
          sml::state<applying_update>,
      sml::state<prepare_update_decision> [guard::prepare_update_no_update{}] =
          sml::state<publishing_update>,
      sml::state<prepare_update_decision> [guard::prepare_update_invalid_status{}] /
          action::set_backend_error = sml::state<errored>,

      sml::state<preparing_batch> / action::run_prepare_batch_phase =
          sml::state<prepare_batch_decision>,
      sml::state<prepare_batch_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<prepare_batch_decision> [guard::phase_ok{}] =
          sml::state<publishing_batch>,

      sml::state<preparing_full> / action::run_prepare_full_phase =
          sml::state<prepare_full_decision>,
      sml::state<prepare_full_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<prepare_full_decision> [guard::phase_ok{}] =
          sml::state<publishing_full>,

      sml::state<applying_update> [guard::apply_update_ready{}] /
          action::run_apply_update_phase = sml::state<apply_update_decision>,
      sml::state<applying_update> [guard::apply_update_backend_failed{}] /
          action::set_backend_error = sml::state<errored>,
      sml::state<applying_update> [guard::apply_update_invalid_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<apply_update_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<apply_update_decision> [guard::phase_ok{}] =
          sml::state<publishing_update>,

      sml::state<publishing_update> [guard::valid_publish_update_context{}] /
          action::run_publish_update_phase = sml::state<publish_update_decision>,
      sml::state<publishing_update> [guard::invalid_publish_update_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<publish_update_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_update_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<publishing_batch> [guard::valid_publish_batch_context{}] /
          action::run_publish_batch_phase = sml::state<publish_batch_decision>,
      sml::state<publishing_batch> [guard::invalid_publish_batch_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<publish_batch_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_batch_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<publishing_full> [guard::valid_publish_full_context{}] /
          action::run_publish_full_phase = sml::state<publish_full_decision>,
      sml::state<publishing_full> [guard::invalid_publish_full_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<publish_full_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_full_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::mark_done = sml::state<initialized>,
      sml::state<errored> / action::ensure_last_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<validating_update> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<validating_batch> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<validating_full> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<preparing_update> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<prepare_update_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<preparing_batch> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<prepare_batch_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<preparing_full> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<prepare_full_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<applying_update> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<apply_update_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publishing_update> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publish_update_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publishing_batch> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publish_batch_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publishing_full> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<publish_full_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::prepare_update & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK && ev.status_out != nullptr) {
      *ev.status_out = context_.prepared_status;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::prepare_batch & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK && ev.status_out != nullptr) {
      *ev.status_out = context_.prepared_status;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::prepare_full & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK && ev.status_out != nullptr) {
      *ev.status_out = context_.prepared_status;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }
  event::memory_status last_status() const noexcept { return context_.prepared_status; }

 private:
  using base_type::raw_sm;

  action::context context_{};
};

}  // namespace emel::memory::coordinator
