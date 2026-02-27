#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "emel/memory/recurrent/context.hpp"
#include "emel/memory/recurrent/detail.hpp"
#include "emel/memory/recurrent/events.hpp"

namespace emel::memory::recurrent::action {

namespace detail {

inline int32_t resolve_positive_or_default(const int32_t value, const int32_t fallback) noexcept {
  const int32_t use_value = static_cast<int32_t>(value > 0);
  return use_value * value + (1 - use_value) * fallback;
}

inline void reset_runtime(context & ctx) noexcept {
  ctx.slots.reset();
  ctx.seq_to_slot.fill(recurrent::detail::invalid_slot);
  ctx.slot_owner_seq.fill(recurrent::detail::invalid_slot);
  ctx.sequence_length.fill(0);
  ctx.free_count = ctx.max_slots;
  for (int32_t i = 0; i < ctx.max_slots; ++i) {
    ctx.free_stack[static_cast<size_t>(i)] = ctx.max_slots - 1 - i;
  }
}

inline void fill_snapshot(const context & ctx, view::snapshot & snapshot) noexcept {
  snapshot = view::snapshot{};
  snapshot.max_sequences = ctx.max_sequences;
  snapshot.block_tokens = 16;

  for (int32_t seq_id = 0; seq_id < ctx.max_sequences; ++seq_id) {
    const size_t seq_index = static_cast<size_t>(seq_id);
    const int32_t active =
        static_cast<int32_t>(ctx.seq_to_slot[seq_index] != recurrent::detail::invalid_slot);
    snapshot.sequence_active[seq_index] = static_cast<uint8_t>(active);
    snapshot.sequence_length_values[seq_index] = ctx.sequence_length[seq_index] * active;
    snapshot.sequence_recurrent_slot[seq_index] =
        active * ctx.seq_to_slot[seq_index] + (1 - active) * recurrent::detail::invalid_slot;
  }
}

}  // namespace detail

struct begin_reserve {
  void operator()(const event::reserve_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.resolved_max_sequences = 0;
    ev.ctx.resolved_slots = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_allocate_sequence {
  void operator()(const event::allocate_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.slot_id = recurrent::detail::invalid_slot;
    ev.ctx.slot_activated = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_allocate_slots {
  void operator()(const event::allocate_slots_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.block_count = 0;
    ev.ctx.old_length = 0;
    ev.ctx.new_length = 0;
    ev.block_count_out = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_branch_sequence {
  void operator()(const event::branch_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.child_slot = recurrent::detail::invalid_slot;
    ev.ctx.slot_activated = false;
    ev.ctx.copy_accepted = false;
    ev.ctx.copy_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_free_sequence {
  void operator()(const event::free_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.slot_id = recurrent::detail::invalid_slot;
    ev.ctx.slot_deactivated = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_rollback_slots {
  void operator()(const event::rollback_slots_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.block_count = 0;
    ev.ctx.current_length = 0;
    ev.ctx.new_length = 0;
    ev.block_count_out = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_capture_view {
  void operator()(const event::capture_view_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct mark_operation_success {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.accepted = true;
    runtime_ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_reserve {
  void operator()(const event::reserve_runtime & ev, context & ctx) const noexcept {
    ev.ctx.resolved_max_sequences = detail::resolve_positive_or_default(
        ev.request.max_sequences, recurrent::detail::max_sequences);
    const int32_t requested_slots =
        detail::resolve_positive_or_default(ev.request.max_blocks, ev.ctx.resolved_max_sequences);
    ev.ctx.resolved_slots = std::min(ev.ctx.resolved_max_sequences, requested_slots);

    ctx.max_sequences = ev.ctx.resolved_max_sequences;
    ctx.max_slots = ev.ctx.resolved_slots;
    detail::reset_runtime(ctx);
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_allocate_sequence_inactive {
  void operator()(const event::allocate_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.slot_id = ctx.free_stack[static_cast<size_t>(ctx.free_count - 1)];
    ctx.free_count -= 1;
    ev.ctx.slot_activated =
        ctx.slots.process_indexed<recurrent::detail::slot_activate>(static_cast<size_t>(ev.ctx.slot_id));
    const int32_t activated = static_cast<int32_t>(ev.ctx.slot_activated);

    ctx.free_count += (1 - activated);
    ctx.seq_to_slot[seq_index] =
        activated * ev.ctx.slot_id + (1 - activated) * ctx.seq_to_slot[seq_index];
    ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.slot_id)] =
        activated * ev.request.seq_id +
        (1 - activated) * ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.slot_id)];
    ctx.sequence_length[seq_index] = (1 - activated) * ctx.sequence_length[seq_index];
    ev.ctx.accepted = activated != 0;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_allocate_slots {
  void operator()(const event::allocate_slots_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.old_length = ctx.sequence_length[seq_index];
    ev.ctx.new_length = ev.ctx.old_length + ev.request.token_count;
    ctx.sequence_length[seq_index] = ev.ctx.new_length;
    ev.ctx.block_count = 0;
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_branch_sequence_prepare_child_slot {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.child_slot = ctx.free_stack[static_cast<size_t>(ctx.free_count - 1)];
    ctx.free_count -= 1;
    ev.ctx.slot_activated = ctx.slots.process_indexed<recurrent::detail::slot_activate>(
        static_cast<size_t>(ev.ctx.child_slot));
    const int32_t activated = static_cast<int32_t>(ev.ctx.slot_activated);
    ctx.free_count += (1 - activated);
  }
};

struct exec_branch_sequence_copy_callback {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t parent_index = static_cast<size_t>(ev.request.parent_seq_id);
    const int32_t parent_slot = ctx.seq_to_slot[parent_index];
    ev.ctx.copy_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.copy_accepted = ev.request.copy_state(
        parent_slot,
        ev.ctx.child_slot,
        ev.request.copy_state_user_data,
        &ev.ctx.copy_error);

    const int32_t copy_error_is_none = static_cast<int32_t>(
        ev.ctx.copy_error == static_cast<int32_t>(emel::error::cast(error::none)));
    const int32_t copy_success = static_cast<int32_t>(ev.ctx.copy_accepted) * copy_error_is_none;
    const int32_t copy_failed_with_error = (1 - copy_success) * (1 - copy_error_is_none);
    ev.ctx.accepted = copy_success != 0;
    ev.ctx.operation_error = static_cast<emel::error::type>(
        copy_failed_with_error * ev.ctx.copy_error);
  }
};

struct exec_branch_sequence_rollback_child_slot {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    const bool deactivated = ctx.slots.process_indexed<recurrent::detail::slot_deactivate>(
        static_cast<size_t>(ev.ctx.child_slot));
    const int32_t deactivated_int = static_cast<int32_t>(deactivated);
    ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.child_slot)] =
        deactivated_int * recurrent::detail::invalid_slot +
        (1 - deactivated_int) * ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.child_slot)];
    ctx.free_stack[static_cast<size_t>(ctx.free_count)] = ev.ctx.child_slot;
    ctx.free_count += deactivated_int;
  }
};

struct finalize_branch_sequence_success {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t parent_index = static_cast<size_t>(ev.request.parent_seq_id);
    const size_t child_index = static_cast<size_t>(ev.request.child_seq_id);
    ctx.seq_to_slot[child_index] = ev.ctx.child_slot;
    ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.child_slot)] = ev.request.child_seq_id;
    ctx.sequence_length[child_index] = ctx.sequence_length[parent_index];
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_free_sequence_active {
  void operator()(const event::free_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.slot_id = ctx.seq_to_slot[seq_index];
    ev.ctx.slot_deactivated = ctx.slots.process_indexed<recurrent::detail::slot_deactivate>(
        static_cast<size_t>(ev.ctx.slot_id));
    const int32_t deactivated = static_cast<int32_t>(ev.ctx.slot_deactivated);

    ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.slot_id)] =
        deactivated * recurrent::detail::invalid_slot +
        (1 - deactivated) * ctx.slot_owner_seq[static_cast<size_t>(ev.ctx.slot_id)];
    ctx.seq_to_slot[seq_index] =
        deactivated * recurrent::detail::invalid_slot + (1 - deactivated) * ctx.seq_to_slot[seq_index];
    ctx.sequence_length[seq_index] = (1 - deactivated) * ctx.sequence_length[seq_index];
    ctx.free_stack[static_cast<size_t>(ctx.free_count)] = ev.ctx.slot_id;
    ctx.free_count += deactivated;

    ev.ctx.accepted = deactivated != 0;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_rollback_slots {
  void operator()(const event::rollback_slots_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.current_length = ctx.sequence_length[seq_index];
    ev.ctx.new_length = std::max(0, ev.ctx.current_length - ev.request.token_count);
    ctx.sequence_length[seq_index] = ev.ctx.new_length;
    ev.ctx.block_count = 0;
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_capture_view {
  void operator()(const event::capture_view_runtime & ev, context & ctx) const noexcept {
    detail::fill_snapshot(ctx, ev.snapshot_out);
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_operation {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = recurrent::detail::cast_api_error(runtime_ev.ctx.operation_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
    if constexpr (requires { runtime_ev.block_count_out; runtime_ev.ctx.block_count; }) {
      runtime_ev.block_count_out = runtime_ev.ctx.block_count;
    }
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = recurrent::detail::unwrap_runtime_event(ev);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
    if constexpr (requires { runtime_ev.block_count_out; }) {
      runtime_ev.block_count_out = 0;
    }
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      ev.error_code_out = static_cast<int32_t>(ev.ctx.err);
      if constexpr (requires { ev.block_count_out; }) {
        ev.block_count_out = 0;
      }
    }
  }
};

inline constexpr begin_reserve begin_reserve{};
inline constexpr begin_allocate_sequence begin_allocate_sequence{};
inline constexpr begin_allocate_slots begin_allocate_slots{};
inline constexpr begin_branch_sequence begin_branch_sequence{};
inline constexpr begin_free_sequence begin_free_sequence{};
inline constexpr begin_rollback_slots begin_rollback_slots{};
inline constexpr begin_capture_view begin_capture_view{};
inline constexpr mark_operation_success mark_operation_success{};
inline constexpr exec_reserve exec_reserve{};
inline constexpr exec_allocate_sequence_inactive exec_allocate_sequence_inactive{};
inline constexpr exec_allocate_slots exec_allocate_slots{};
inline constexpr exec_branch_sequence_prepare_child_slot exec_branch_sequence_prepare_child_slot{};
inline constexpr exec_branch_sequence_copy_callback exec_branch_sequence_copy_callback{};
inline constexpr exec_branch_sequence_rollback_child_slot exec_branch_sequence_rollback_child_slot{};
inline constexpr finalize_branch_sequence_success finalize_branch_sequence_success{};
inline constexpr exec_free_sequence_active exec_free_sequence_active{};
inline constexpr exec_rollback_slots exec_rollback_slots{};
inline constexpr exec_capture_view exec_capture_view{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_error_from_operation mark_error_from_operation{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::memory::recurrent::action
