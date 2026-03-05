#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/memory/recurrent/context.hpp"
#include "emel/memory/recurrent/detail.hpp"
#include "emel/memory/recurrent/events.hpp"

namespace emel::memory::recurrent::guard {

struct reserve_request_valid {
  bool operator()(const event::reserve_runtime & ev) const noexcept {
    const int32_t max_sequence_count =
        ev.request.max_sequences > 0 ? ev.request.max_sequences : recurrent::detail::max_sequences;
    const int32_t requested_slots =
        ev.request.max_blocks > 0 ? ev.request.max_blocks : max_sequence_count;

    return max_sequence_count > 0 && max_sequence_count <= recurrent::detail::max_sequences &&
           requested_slots > 0;
  }
};

struct reserve_request_invalid {
  bool operator()(const event::reserve_runtime & ev) const noexcept {
    return !reserve_request_valid{}(ev);
  }
};

struct allocate_sequence_request_valid {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id);
  }
};

struct allocate_sequence_request_invalid {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_sequence_request_valid{}(ev, ctx);
  }
};

struct allocate_sequence_target_active {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] != recurrent::detail::invalid_slot;
  }
};

struct allocate_sequence_request_active {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return allocate_sequence_request_valid{}(ev, ctx) &&
           allocate_sequence_target_active{}(ev, ctx);
  }
};

struct allocate_sequence_target_inactive_with_slot {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] == recurrent::detail::invalid_slot &&
           ctx.free_count > 0;
  }
};

struct allocate_sequence_request_inactive_with_slot {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return allocate_sequence_request_valid{}(ev, ctx) &&
           allocate_sequence_target_inactive_with_slot{}(ev, ctx);
  }
};

struct allocate_sequence_target_inactive_without_slot {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] == recurrent::detail::invalid_slot &&
           ctx.free_count <= 0;
  }
};

struct allocate_sequence_request_inactive_without_slot {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return allocate_sequence_request_valid{}(ev, ctx) &&
           allocate_sequence_target_inactive_without_slot{}(ev, ctx);
  }
};

struct allocate_slots_request_shape_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const bool seq_id_valid =
        recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id);
    const int32_t safe_seq_id = static_cast<int32_t>(seq_id_valid) * ev.request.seq_id;
    const size_t seq_index = static_cast<size_t>(safe_seq_id);
    const bool seq_active = ctx.seq_to_slot[seq_index] != recurrent::detail::invalid_slot;
    return seq_id_valid && ev.request.token_count > 0 && seq_active;
  }
};

struct allocate_slots_request_shape_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_slots_request_shape_valid{}(ev, ctx);
  }
};

struct allocate_slots_request_length_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const bool shape_valid = allocate_slots_request_shape_valid{}(ev, ctx);
    const int32_t safe_seq_id = static_cast<int32_t>(shape_valid) * ev.request.seq_id;
    const size_t seq_index = static_cast<size_t>(safe_seq_id);
    const int64_t new_length_wide =
        static_cast<int64_t>(ctx.sequence_length[seq_index]) + ev.request.token_count;
    return shape_valid && new_length_wide >= 0 &&
           new_length_wide <= std::numeric_limits<int32_t>::max();
  }
};

struct allocate_slots_request_length_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_slots_request_length_valid{}(ev, ctx);
  }
};

struct branch_sequence_request_shape_valid {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    const bool has_copy_callback = ev.request.copy_state != nullptr;
    const bool parent_id_valid =
        recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.parent_seq_id);
    const bool child_id_valid =
        recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.child_seq_id);
    const bool ids_distinct = ev.request.parent_seq_id != ev.request.child_seq_id;
    const int32_t safe_parent_id = static_cast<int32_t>(parent_id_valid) * ev.request.parent_seq_id;
    const int32_t safe_child_id = static_cast<int32_t>(child_id_valid) * ev.request.child_seq_id;
    const size_t parent_index = static_cast<size_t>(safe_parent_id);
    const size_t child_index = static_cast<size_t>(safe_child_id);
    const bool parent_active = ctx.seq_to_slot[parent_index] != recurrent::detail::invalid_slot;
    const bool child_inactive = ctx.seq_to_slot[child_index] == recurrent::detail::invalid_slot;
    return has_copy_callback && parent_id_valid && child_id_valid && ids_distinct &&
           parent_active && child_inactive;
  }
};

struct branch_sequence_request_shape_invalid {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !branch_sequence_request_shape_valid{}(ev, ctx);
  }
};

struct branch_sequence_request_capacity_available {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return branch_sequence_request_shape_valid{}(ev, ctx) && ctx.free_count > 0;
  }
};

struct branch_sequence_request_capacity_exhausted {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return branch_sequence_request_shape_valid{}(ev, ctx) && ctx.free_count <= 0;
  }
};

struct branch_slot_activation_succeeded {
  bool operator()(const event::branch_sequence_runtime & ev) const noexcept {
    return ev.ctx.slot_activated;
  }
};

struct branch_slot_activation_failed {
  bool operator()(const event::branch_sequence_runtime & ev) const noexcept {
    return !ev.ctx.slot_activated;
  }
};

struct branch_copy_succeeded {
  bool operator()(const event::branch_sequence_runtime & ev) const noexcept {
    return ev.ctx.copy_accepted &&
           ev.ctx.copy_error == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct branch_copy_failed_with_error {
  bool operator()(const event::branch_sequence_runtime & ev) const noexcept {
    return ev.ctx.copy_error != static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct branch_copy_failed_without_error {
  bool operator()(const event::branch_sequence_runtime & ev) const noexcept {
    return !ev.ctx.copy_accepted &&
           ev.ctx.copy_error == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct free_sequence_request_valid {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id);
  }
};

struct free_sequence_request_invalid {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !free_sequence_request_valid{}(ev, ctx);
  }
};

struct free_sequence_target_active {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] != recurrent::detail::invalid_slot;
  }
};

struct free_sequence_request_active {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return free_sequence_request_valid{}(ev, ctx) && free_sequence_target_active{}(ev, ctx);
  }
};

struct free_sequence_target_inactive {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] == recurrent::detail::invalid_slot;
  }
};

struct free_sequence_request_inactive {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return free_sequence_request_valid{}(ev, ctx) && free_sequence_target_inactive{}(ev, ctx);
  }
};

struct rollback_slots_request_valid {
  bool operator()(const event::rollback_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return recurrent::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id) &&
           ev.request.token_count > 0 &&
           ctx.seq_to_slot[static_cast<size_t>(ev.request.seq_id)] != recurrent::detail::invalid_slot;
  }
};

struct rollback_slots_request_invalid {
  bool operator()(const event::rollback_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !rollback_slots_request_valid{}(ev, ctx);
  }
};

struct capture_request_valid {
  bool operator()(const event::capture_view_runtime & ev) const noexcept {
    return ev.has_snapshot_out;
  }
};

struct capture_request_invalid {
  bool operator()(const event::capture_view_runtime & ev) const noexcept {
    return !capture_request_valid{}(ev);
  }
};

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return recurrent::detail::unwrap_runtime_event(ev).ctx.accepted;
  }
};

struct operation_failed_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.accepted &&
           runtime_ev.ctx.operation_error != emel::error::cast(error::none);
  }
};

struct operation_failed_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.accepted &&
           runtime_ev.ctx.operation_error == emel::error::cast(error::none);
  }
};

}  // namespace emel::memory::recurrent::guard
