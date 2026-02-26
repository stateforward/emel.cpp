#pragma once

// benchmark: scaffold

#include <memory>
#include <cstdint>

#include "emel/memory/recurrent/actions.hpp"
#include "emel/memory/recurrent/events.hpp"
#include "emel/memory/recurrent/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::recurrent {

struct uninitialized {};
struct initializing {};
struct ready {};
struct allocating_sequence {};
struct allocating_slots {};
struct branching_sequence {};
struct freeing_sequence {};
struct rolling_back_slots {};
struct out_of_memory {};
struct errored {};
struct unexpected {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<uninitialized> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,
        sml::state<ready> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,
        sml::state<unexpected> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,

        sml::state<initializing>[guard::phase_ok] = sml::state<ready>,
        sml::state<initializing>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<initializing>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::allocate_sequence> / action::begin_allocate_sequence =
            sml::state<allocating_sequence>,
        sml::state<allocating_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<allocating_sequence>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<allocating_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::allocate_slots> / action::begin_allocate_slots =
            sml::state<allocating_slots>,
        sml::state<allocating_slots>[guard::phase_ok] = sml::state<ready>,
        sml::state<allocating_slots>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::branch_sequence> / action::begin_branch_sequence =
            sml::state<branching_sequence>,
        sml::state<branching_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<branching_sequence>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<branching_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::free_sequence> / action::begin_free_sequence =
            sml::state<freeing_sequence>,
        sml::state<freeing_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<freeing_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::rollback_slots> / action::begin_rollback_slots =
            sml::state<rolling_back_slots>,
        sml::state<rolling_back_slots>[guard::phase_ok] = sml::state<ready>,
        sml::state<rolling_back_slots>[guard::phase_failed] = sml::state<errored>,

        sml::state<out_of_memory> / action::clear_out_of_memory = sml::state<ready>,
        sml::state<errored> / action::ensure_last_error = sml::state<ready>,

        sml::state<uninitialized> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<uninitialized>,
        sml::state<initializing> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<initializing>,
        sml::state<ready> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<ready>,
        sml::state<allocating_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<allocating_sequence>,
        sml::state<allocating_slots> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<allocating_slots>,
        sml::state<branching_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<branching_sequence>,
        sml::state<freeing_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<freeing_sequence>,
        sml::state<rolling_back_slots> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<rolling_back_slots>,
        sml::state<out_of_memory> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<out_of_memory>,
        sml::state<errored> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<errored>,
        sml::state<unexpected> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<unexpected>,

        sml::state<uninitialized> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<initializing> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<ready> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<allocating_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<allocating_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<branching_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<freeing_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<rolling_back_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<out_of_memory> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  // One-time heap storage keeps snapshot handoff frozen without per-dispatch allocation.
  sm() : base_type(context_), snapshot_(std::make_unique<view::snapshot>()) {}

  bool process_event(const event::reserve & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_slots & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::branch_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::free_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::rollback_slots & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::capture_view & ev) { return base_type::process_event(ev); }

  using base_type::process_event;

  int32_t last_error() const noexcept { return context_.last_error; }

  bool is_sequence_active(const int32_t seq_id) const noexcept {
    return action::is_active(context_, seq_id);
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    return action::sequence_length_value(context_, seq_id);
  }

  int32_t lookup_kv_block(const int32_t, const int32_t) const noexcept { return -1; }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    return action::lookup_slot(context_, seq_id);
  }

  view::any view() noexcept {
    if (snapshot_ == nullptr) {
      return view::any{};
    }
    int32_t err = EMEL_OK;
    (void)this->base_type::process_event(event::capture_view{
      .snapshot_out = snapshot_.get(),
      .error_out = &err,
    });
    if (err != EMEL_OK) {
      return view::any{};
    }

    return view::any{.frozen = snapshot_.get()};
  }

 private:
  template <class ev>
  bool process_lifecycle_event(const ev & event) {
    const bool accepted = base_type::process_event(event);
    if constexpr (requires { event.error_out; }) {
      if (event.error_out != nullptr) {
        *event.error_out = context_.last_error;
      }
    }
    return accepted && context_.last_error == EMEL_OK;
  }

  std::unique_ptr<view::snapshot> snapshot_;
  action::context context_{};
};

}  // namespace emel::memory::recurrent
