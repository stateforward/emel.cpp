#pragma once
// benchmark: scaffold

#include <algorithm>
#include <cstdint>

#include "emel/memory/hybrid/actions.hpp"
#include "emel/memory/hybrid/events.hpp"
#include "emel/memory/hybrid/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::hybrid {

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

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<uninitialized> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,
        sml::state<ready> + sml::event<event::reserve> / action::begin_reserve =
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
        sml::state<allocating_slots>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
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

        sml::state<uninitialized> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<initializing> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<ready> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<allocating_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<allocating_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<branching_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<freeing_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<rolling_back_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<out_of_memory> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>,
        sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<errored>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::reserve & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_slots & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::branch_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::free_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::rollback_slots & ev) { return process_lifecycle_event(ev); }

  using base_type::process_event;

  int32_t last_error() const noexcept { return context_.last_error; }

  bool is_sequence_active(const int32_t seq_id) const noexcept {
    return context_.kv.is_sequence_active(seq_id) && context_.recurrent.is_sequence_active(seq_id);
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    return std::min(context_.kv.sequence_length(seq_id), context_.recurrent.sequence_length(seq_id));
  }

  int32_t lookup_kv_block(const int32_t seq_id, const int32_t pos) const noexcept {
    return context_.kv.lookup_kv_block(seq_id, pos);
  }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    return context_.recurrent.lookup_recurrent_slot(seq_id);
  }

  view::any view() const noexcept {
    return view::any{
        .self = this,
        .is_sequence_active_impl = &is_sequence_active_thunk,
        .sequence_length_impl = &sequence_length_thunk,
        .lookup_kv_block_impl = &lookup_kv_block_thunk,
        .lookup_recurrent_slot_impl = &lookup_recurrent_slot_thunk,
    };
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

  static bool is_sequence_active_thunk(const void * self, const int32_t seq_id) {
    return static_cast<const sm *>(self)->is_sequence_active(seq_id);
  }

  static int32_t sequence_length_thunk(const void * self, const int32_t seq_id) {
    return static_cast<const sm *>(self)->sequence_length(seq_id);
  }

  static int32_t lookup_kv_block_thunk(const void * self, const int32_t seq_id,
                                       const int32_t pos) {
    return static_cast<const sm *>(self)->lookup_kv_block(seq_id, pos);
  }

  static int32_t lookup_recurrent_slot_thunk(const void * self, const int32_t seq_id) {
    return static_cast<const sm *>(self)->lookup_recurrent_slot(seq_id);
  }

  action::context context_{};
};

}  // namespace emel::memory::hybrid
