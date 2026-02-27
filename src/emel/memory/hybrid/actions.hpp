#pragma once

#include <algorithm>

#include "emel/memory/hybrid/context.hpp"
#include "emel/memory/hybrid/detail.hpp"
#include "emel/memory/hybrid/events.hpp"

namespace emel::memory::hybrid::action {

namespace detail {

inline void merge_snapshots(const view::snapshot & kv_snapshot,
                            const view::snapshot & recurrent_snapshot,
                            view::snapshot & out) noexcept {
  out = view::snapshot{};
  out.max_sequences = std::min(kv_snapshot.max_sequences, recurrent_snapshot.max_sequences);
  out.block_tokens = kv_snapshot.block_tokens;

  for (int32_t seq_id = 0; seq_id < out.max_sequences; ++seq_id) {
    const size_t seq = static_cast<size_t>(seq_id);
    const int32_t active = static_cast<int32_t>(
        (kv_snapshot.sequence_active[seq] != 0u) &&
        (recurrent_snapshot.sequence_active[seq] != 0u));
    const int32_t inactive = 1 - active;

    out.sequence_active[seq] = static_cast<uint8_t>(active);
    out.sequence_length_values[seq] =
        std::min(kv_snapshot.sequence_length_values[seq],
                 recurrent_snapshot.sequence_length_values[seq]) *
        active;
    out.sequence_kv_block_count[seq] = kv_snapshot.sequence_kv_block_count[seq] * active;
    out.sequence_kv_blocks[seq] = kv_snapshot.sequence_kv_blocks[seq];
    out.sequence_recurrent_slot[seq] =
        recurrent_snapshot.sequence_recurrent_slot[seq] * active + (-1 * inactive);
  }
}

}  // namespace detail

struct begin_reserve {
  void operator()(const event::reserve_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_allocate_sequence {
  void operator()(const event::allocate_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.rollback_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_allocate_slots {
  void operator()(const event::allocate_slots_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.rollback_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_block_count = 0;
    ev.block_count_out = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_branch_sequence {
  void operator()(const event::branch_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.rollback_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_free_sequence {
  void operator()(const event::free_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_rollback_slots {
  void operator()(const event::rollback_slots_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_block_count = 0;
    ev.block_count_out = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_capture_view {
  void operator()(const event::capture_view_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.kv_accepted = false;
    ev.ctx.recurrent_accepted = false;
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct exec_reserve_kv {
  void operator()(const event::reserve_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_accepted = ctx.kv.process_event(event::reserve{
      .max_sequences = ev.request.max_sequences,
      .max_blocks = ev.request.max_blocks,
      .block_tokens = ev.request.block_tokens,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_reserve_recurrent {
  void operator()(const event::reserve_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::reserve{
      .max_sequences = ev.request.max_sequences,
      .max_blocks = ev.request.max_blocks,
      .block_tokens = ev.request.block_tokens,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_allocate_sequence_kv {
  void operator()(const event::allocate_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_accepted = ctx.kv.process_event(event::allocate_sequence{
      .seq_id = ev.request.seq_id,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_allocate_sequence_recurrent {
  void operator()(const event::allocate_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::allocate_sequence{
      .seq_id = ev.request.seq_id,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_allocate_sequence_rollback_kv {
  void operator()(const event::allocate_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_accepted = ctx.kv.process_event(event::free_sequence{
      .seq_id = ev.request.seq_id,
      .error_out = &ev.ctx.rollback_error,
    });
  }
};

struct exec_allocate_slots_kv {
  void operator()(const event::allocate_slots_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_block_count = 0;
    ev.ctx.kv_accepted = ctx.kv.process_event(event::allocate_slots{
      .seq_id = ev.request.seq_id,
      .token_count = ev.request.token_count,
      .block_count_out = &ev.ctx.kv_block_count,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_allocate_slots_recurrent {
  void operator()(const event::allocate_slots_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::allocate_slots{
      .seq_id = ev.request.seq_id,
      .token_count = ev.request.token_count,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_allocate_slots_rollback_kv {
  void operator()(const event::allocate_slots_runtime & ev, context & ctx) const noexcept {
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_accepted = ctx.kv.process_event(event::rollback_slots{
      .seq_id = ev.request.seq_id,
      .token_count = ev.request.token_count,
      .error_out = &ev.ctx.rollback_error,
    });
  }
};

struct exec_branch_sequence_kv {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_accepted = ctx.kv.process_event(event::branch_sequence{
      .parent_seq_id = ev.request.parent_seq_id,
      .child_seq_id = ev.request.child_seq_id,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_branch_sequence_recurrent {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::branch_sequence{
      .parent_seq_id = ev.request.parent_seq_id,
      .child_seq_id = ev.request.child_seq_id,
      .copy_state = ev.request.copy_state,
      .copy_state_user_data = ev.request.copy_state_user_data,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_branch_sequence_rollback_kv {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.rollback_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.rollback_accepted = ctx.kv.process_event(event::free_sequence{
      .seq_id = ev.request.child_seq_id,
      .error_out = &ev.ctx.rollback_error,
    });
  }
};

struct exec_free_sequence_kv {
  void operator()(const event::free_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_accepted = ctx.kv.process_event(event::free_sequence{
      .seq_id = ev.request.seq_id,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_free_sequence_recurrent {
  void operator()(const event::free_sequence_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::free_sequence{
      .seq_id = ev.request.seq_id,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_rollback_slots_kv {
  void operator()(const event::rollback_slots_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_block_count = 0;
    ev.ctx.kv_accepted = ctx.kv.process_event(event::rollback_slots{
      .seq_id = ev.request.seq_id,
      .token_count = ev.request.token_count,
      .block_count_out = &ev.ctx.kv_block_count,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_rollback_slots_recurrent {
  void operator()(const event::rollback_slots_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::rollback_slots{
      .seq_id = ev.request.seq_id,
      .token_count = ev.request.token_count,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct exec_capture_kv {
  void operator()(const event::capture_view_runtime & ev, context & ctx) const noexcept {
    ev.ctx.kv_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.kv_accepted = ctx.kv.process_event(event::capture_view{
      .snapshot_out = &ev.kv_snapshot,
      .error_out = &ev.ctx.kv_error,
    });
  }
};

struct exec_capture_recurrent {
  void operator()(const event::capture_view_runtime & ev, context & ctx) const noexcept {
    ev.ctx.recurrent_error = static_cast<int32_t>(emel::error::cast(error::none));
    ev.ctx.recurrent_accepted = ctx.recurrent.process_event(event::capture_view{
      .snapshot_out = &ev.recurrent_snapshot,
      .error_out = &ev.ctx.recurrent_error,
    });
  }
};

struct merge_capture_snapshots {
  void operator()(const event::capture_view_runtime & ev, context &) const noexcept {
    detail::merge_snapshots(ev.kv_snapshot, ev.recurrent_snapshot, ev.snapshot_out);
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_out_of_memory {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::out_of_memory);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_kv {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = hybrid::detail::cast_api_error(runtime_ev.ctx.kv_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_recurrent {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = hybrid::detail::cast_api_error(runtime_ev.ctx.recurrent_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_rollback {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = hybrid::detail::cast_api_error(runtime_ev.ctx.rollback_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
    if constexpr (requires { runtime_ev.block_count_out; runtime_ev.ctx.kv_block_count; }) {
      runtime_ev.block_count_out = runtime_ev.ctx.kv_block_count;
    }
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
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
inline constexpr exec_reserve_kv exec_reserve_kv{};
inline constexpr exec_reserve_recurrent exec_reserve_recurrent{};
inline constexpr exec_allocate_sequence_kv exec_allocate_sequence_kv{};
inline constexpr exec_allocate_sequence_recurrent exec_allocate_sequence_recurrent{};
inline constexpr exec_allocate_sequence_rollback_kv exec_allocate_sequence_rollback_kv{};
inline constexpr exec_allocate_slots_kv exec_allocate_slots_kv{};
inline constexpr exec_allocate_slots_recurrent exec_allocate_slots_recurrent{};
inline constexpr exec_allocate_slots_rollback_kv exec_allocate_slots_rollback_kv{};
inline constexpr exec_branch_sequence_kv exec_branch_sequence_kv{};
inline constexpr exec_branch_sequence_recurrent exec_branch_sequence_recurrent{};
inline constexpr exec_branch_sequence_rollback_kv exec_branch_sequence_rollback_kv{};
inline constexpr exec_free_sequence_kv exec_free_sequence_kv{};
inline constexpr exec_free_sequence_recurrent exec_free_sequence_recurrent{};
inline constexpr exec_rollback_slots_kv exec_rollback_slots_kv{};
inline constexpr exec_rollback_slots_recurrent exec_rollback_slots_recurrent{};
inline constexpr exec_capture_kv exec_capture_kv{};
inline constexpr exec_capture_recurrent exec_capture_recurrent{};
inline constexpr merge_capture_snapshots merge_capture_snapshots{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_out_of_memory mark_out_of_memory{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr mark_error_from_kv mark_error_from_kv{};
inline constexpr mark_error_from_recurrent mark_error_from_recurrent{};
inline constexpr mark_error_from_rollback mark_error_from_rollback{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::memory::hybrid::action
