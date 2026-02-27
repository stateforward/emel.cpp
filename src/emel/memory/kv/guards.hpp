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
  if (token_count <= 0 || block_tokens <= 0) {
    return 0;
  }
  return (token_count + block_tokens - 1) / block_tokens;
}

struct allocate_slots_analysis {
  bool request_shape_valid = false;
  bool length_valid = false;
  bool block_layout_valid = false;
  bool capacity_valid = false;
};

inline allocate_slots_analysis analyze_allocate_slots_request(const action::context & ctx,
                                                             const event::allocate_slots & request) noexcept {
  allocate_slots_analysis analysis{};
  analysis.request_shape_valid = kv::detail::valid_sequence_id(ctx.max_sequences, request.seq_id) &&
                                 request.token_count > 0 && ctx.block_tokens > 0;
  if (!analysis.request_shape_valid) {
    return analysis;
  }

  const size_t seq_index = static_cast<size_t>(request.seq_id);
  if (!ctx.sequence_active[seq_index]) {
    analysis.request_shape_valid = false;
    return analysis;
  }

  const int32_t old_length = ctx.sequence_length[seq_index];
  const int64_t new_length_wide = static_cast<int64_t>(old_length) + request.token_count;
  analysis.length_valid =
      new_length_wide > 0 && new_length_wide <= std::numeric_limits<int32_t>::max();
  if (!analysis.length_valid) {
    return analysis;
  }

  const int32_t new_length = static_cast<int32_t>(new_length_wide);
  const int32_t existing_block_count = ctx.sequence_block_count[seq_index];
  const int32_t old_blocks = blocks_for_length(ctx.block_tokens, old_length);
  const int32_t new_blocks = blocks_for_length(ctx.block_tokens, new_length);
  analysis.block_layout_valid = existing_block_count >= old_blocks && new_blocks >= old_blocks;
  if (!analysis.block_layout_valid) {
    return analysis;
  }

  const int32_t blocks_needed = new_blocks - old_blocks;
  const bool within_sequence_capacity =
      existing_block_count + blocks_needed <= kv::detail::max_blocks_per_sequence;
  const bool enough_free_blocks = ctx.free_count >= blocks_needed;
  analysis.capacity_valid = within_sequence_capacity && enough_free_blocks;
  return analysis;
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

struct allocate_slots_request_valid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const detail::allocate_slots_analysis analysis =
        detail::analyze_allocate_slots_request(ctx, ev.request);
    return analysis.request_shape_valid && analysis.length_valid && analysis.block_layout_valid &&
           analysis.capacity_valid;
  }
};

struct allocate_slots_request_invalid {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const detail::allocate_slots_analysis analysis =
        detail::analyze_allocate_slots_request(ctx, ev.request);
    return !analysis.request_shape_valid || !analysis.length_valid;
  }
};

struct allocate_slots_request_backend_error {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const detail::allocate_slots_analysis analysis =
        detail::analyze_allocate_slots_request(ctx, ev.request);
    return analysis.request_shape_valid && analysis.length_valid &&
           !analysis.block_layout_valid;
  }
};

struct allocate_slots_request_out_of_memory {
  bool operator()(const event::allocate_slots_runtime & ev,
                  const action::context & ctx) const noexcept {
    const detail::allocate_slots_analysis analysis =
        detail::analyze_allocate_slots_request(ctx, ev.request);
    return analysis.request_shape_valid && analysis.length_valid &&
           analysis.block_layout_valid && !analysis.capacity_valid;
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
