#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/memory/kv/context.hpp"
#include "emel/memory/kv/detail.hpp"
#include "emel/memory/kv/events.hpp"

namespace emel::memory::kv::guard {

namespace detail {

inline int32_t blocks_for_length(const int32_t block_tokens, const int32_t token_count) noexcept {
  const int32_t positive_tokens = static_cast<int32_t>(token_count > 0);
  const int32_t positive_block_tokens = static_cast<int32_t>(block_tokens > 0);
  const int32_t safe_block_tokens = block_tokens + static_cast<int32_t>(block_tokens <= 0);
  const int32_t effective_tokens = positive_tokens * positive_block_tokens * token_count;
  const int32_t rounded = (effective_tokens + safe_block_tokens - 1) / safe_block_tokens;
  return rounded * positive_block_tokens;
}

}  // namespace detail

struct reserve_request_valid {
  bool operator()(const event::reserve_runtime & ev) const noexcept {
    const int32_t max_sequence_count =
        ev.request.max_sequences > 0 ? ev.request.max_sequences : kv::detail::max_sequences;
    const int32_t max_block_count =
        ev.request.max_blocks > 0 ? ev.request.max_blocks : kv::detail::max_blocks;
    const int32_t block_token_count =
        ev.request.block_tokens > 0 ? ev.request.block_tokens : kv::detail::default_block_tokens;

    return max_sequence_count > 0 && max_sequence_count <= kv::detail::max_sequences &&
           max_block_count > 0 && max_block_count <= kv::detail::max_blocks &&
           block_token_count > 0;
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
    return kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id);
  }
};

struct allocate_sequence_request_invalid {
  bool operator()(const event::allocate_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_sequence_request_valid{}(ev, ctx);
  }
};

struct allocate_slots_request_shape_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id) &&
           ev.request.token_count > 0 && ctx.block_tokens > 0 &&
           ctx.sequence_active[static_cast<size_t>(ev.request.seq_id)];
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
    return shape_valid && new_length_wide > 0 &&
           new_length_wide <= std::numeric_limits<int32_t>::max();
  }
};

struct allocate_slots_request_length_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_slots_request_length_valid{}(ev, ctx);
  }
};

struct allocate_slots_request_block_layout_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const bool length_valid = allocate_slots_request_length_valid{}(ev, ctx);
    const int32_t safe_seq_id = static_cast<int32_t>(length_valid) * ev.request.seq_id;
    const size_t seq_index = static_cast<size_t>(safe_seq_id);
    const int32_t old_length = ctx.sequence_length[seq_index];
    const int32_t new_length = old_length + ev.request.token_count;
    const int32_t existing_block_count = ctx.sequence_block_count[seq_index];
    const int32_t old_blocks = detail::blocks_for_length(ctx.block_tokens, old_length);
    const int32_t new_blocks = detail::blocks_for_length(ctx.block_tokens, new_length);
    return length_valid && existing_block_count >= old_blocks && new_blocks >= old_blocks;
  }
};

struct allocate_slots_request_block_layout_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_slots_request_block_layout_valid{}(ev, ctx);
  }
};

struct allocate_slots_request_capacity_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const bool block_layout_valid = allocate_slots_request_block_layout_valid{}(ev, ctx);
    const int32_t safe_seq_id = static_cast<int32_t>(block_layout_valid) * ev.request.seq_id;
    const size_t seq_index = static_cast<size_t>(safe_seq_id);
    const int32_t old_length = ctx.sequence_length[seq_index];
    const int32_t new_length = old_length + ev.request.token_count;
    const int32_t existing_block_count = ctx.sequence_block_count[seq_index];
    const int32_t old_blocks = detail::blocks_for_length(ctx.block_tokens, old_length);
    const int32_t new_blocks = detail::blocks_for_length(ctx.block_tokens, new_length);
    const int32_t blocks_needed = new_blocks - old_blocks;
    const bool within_sequence_capacity =
        existing_block_count + blocks_needed <= kv::detail::max_blocks_per_sequence;
    const bool enough_free_blocks = ctx.free_count >= blocks_needed;
    return block_layout_valid && within_sequence_capacity && enough_free_blocks;
  }
};

struct allocate_slots_request_capacity_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !allocate_slots_request_capacity_valid{}(ev, ctx);
  }
};

struct branch_sequence_request_valid {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    if (!kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.parent_seq_id) ||
        !kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.child_seq_id) ||
        ev.request.parent_seq_id == ev.request.child_seq_id) {
      return false;
    }

    const size_t parent_index = static_cast<size_t>(ev.request.parent_seq_id);
    const size_t child_index = static_cast<size_t>(ev.request.child_seq_id);
    return ctx.sequence_active[parent_index] && !ctx.sequence_active[child_index];
  }
};

struct branch_sequence_request_invalid {
  bool operator()(const event::branch_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !branch_sequence_request_valid{}(ev, ctx);
  }
};

struct free_sequence_request_valid {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id);
  }
};

struct free_sequence_request_invalid {
  bool operator()(const event::free_sequence_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !free_sequence_request_valid{}(ev, ctx);
  }
};

struct rollback_slots_request_valid {
  bool operator()(const event::rollback_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    return kv::detail::valid_sequence_id(ctx.max_sequences, ev.request.seq_id) &&
           ev.request.token_count > 0 && ctx.block_tokens > 0 &&
           ctx.sequence_active[static_cast<size_t>(ev.request.seq_id)];
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
    return kv::detail::unwrap_runtime_event(ev).ctx.accepted;
  }
};

struct operation_failed_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.accepted &&
           runtime_ev.ctx.operation_error != emel::error::cast(error::none);
  }
};

struct operation_failed_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.accepted &&
           runtime_ev.ctx.operation_error == emel::error::cast(error::none);
  }
};

}  // namespace emel::memory::kv::guard
