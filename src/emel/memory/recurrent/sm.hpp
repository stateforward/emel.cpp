#pragma once

#include <cstdint>

#include "emel/memory/recurrent/actions.hpp"
#include "emel/memory/recurrent/events.hpp"
#include "emel/memory/recurrent/guards.hpp"
#include "emel/sm.hpp"

namespace emel::memory::recurrent {

struct initialized {};
struct ready {};

struct reserving {};
struct reserve_decision {};

struct allocating_sequence {};
struct allocate_sequence_decision {};

struct branching_sequence {};
struct branch_sequence_decision {};

struct freeing_sequence {};
struct free_sequence_decision {};

struct done {};
struct errored {};

/**
 * recurrent memory orchestration model.
 *
 * state purposes:
 * - `initialized`: unreserved lifecycle state.
 * - `ready`: reserved and ready for sequence lifecycle commands.
 * - `reserving`/`reserve_decision`: reserve capacity and reset mappings.
 * - `allocating_sequence`/`branching_sequence`/`freeing_sequence`: bounded
 * lifecycle operations.
 * - `done`/`errored`: terminal outcomes that return to idle states.
 *
 * guard semantics:
 * - guards are pure predicates over `(context)`.
 * - `phase_*` guards branch on `context.phase_error`.
 *
 * action side effects:
 * - actions mutate slot mappings and counters in bounded loops.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::reserve> / action::begin_reserve =
            sml::state<reserving>,
        sml::state<ready> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<reserving>,
        sml::state<reserving>[guard::valid_reserve_context{}] /
            action::run_reserve_phase = sml::state<reserve_decision>,
        sml::state<reserving>[guard::invalid_reserve_context{}] /
            action::set_invalid_argument = sml::state<errored>,
        sml::state<reserve_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<reserve_decision>[guard::phase_ok{}] = sml::state<done>,

        sml::state<ready> + sml::event<event::allocate_sequence> /
                                action::begin_allocate_sequence =
            sml::state<allocating_sequence>,
        sml::state<allocating_sequence>[guard::valid_allocate_context{}] /
            action::run_allocate_phase = sml::state<allocate_sequence_decision>,
        sml::state<allocating_sequence>[guard::invalid_allocate_context{}] /
            action::set_invalid_argument = sml::state<errored>,
        sml::state<allocate_sequence_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<allocate_sequence_decision>[guard::phase_ok{}] =
            sml::state<done>,

        sml::state<ready> +
            sml::event<event::branch_sequence> / action::begin_branch_sequence =
            sml::state<branching_sequence>,
        sml::state<branching_sequence>[guard::valid_branch_context{}] /
            action::run_branch_phase = sml::state<branch_sequence_decision>,
        sml::state<branching_sequence>[guard::invalid_branch_context{}] /
            action::set_invalid_argument = sml::state<errored>,
        sml::state<branch_sequence_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<branch_sequence_decision>[guard::phase_ok{}] =
            sml::state<done>,

        sml::state<ready> +
            sml::event<event::free_sequence> / action::begin_free_sequence =
            sml::state<freeing_sequence>,
        sml::state<freeing_sequence>[guard::valid_free_context{}] /
            action::run_free_phase = sml::state<free_sequence_decision>,
        sml::state<freeing_sequence>[guard::invalid_free_context{}] /
            action::set_invalid_argument = sml::state<errored>,
        sml::state<free_sequence_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<free_sequence_decision>[guard::phase_ok{}] =
            sml::state<done>,

        sml::state<done>[guard::has_capacity{}] / action::mark_done =
            sml::state<ready>,
        sml::state<done>[guard::no_capacity{}] / action::mark_done =
            sml::state<initialized>,
        sml::state<errored>[guard::has_capacity{}] / action::ensure_last_error =
            sml::state<ready>,
        sml::state<errored>[guard::no_capacity{}] / action::ensure_last_error =
            sml::state<initialized>,

        sml::state<initialized> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<ready> + sml::unexpected_event<sml::_> /
                                action::on_unexpected = sml::state<errored>,
        sml::state<reserving> + sml::unexpected_event<sml::_> /
                                    action::on_unexpected = sml::state<errored>,
        sml::state<reserve_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<allocating_sequence> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<allocate_sequence_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<branching_sequence> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<branch_sequence_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<freeing_sequence> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<free_sequence_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<errored>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
                                  action::on_unexpected = sml::state<errored>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::reserve &ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::allocate_sequence &ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::branch_sequence &ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::free_sequence &ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(context_);
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }
  int32_t slot_capacity() const noexcept { return context_.slot_capacity; }
  int32_t active_count() const noexcept { return context_.active_count; }
  int32_t slot_for_sequence(int32_t seq_id) const noexcept {
    return action::slot_for_sequence(context_, seq_id);
  }
  bool has_sequence(int32_t seq_id) const noexcept {
    return action::sequence_exists(context_, seq_id);
  }

private:
  using base_type::raw_sm;

  action::context context_{};
};

} // namespace emel::memory::recurrent
