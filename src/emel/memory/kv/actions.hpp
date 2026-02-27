#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "emel/memory/kv/context.hpp"
#include "emel/memory/kv/detail.hpp"
#include "emel/memory/kv/events.hpp"

namespace emel::memory::kv::action {

namespace detail {

inline int32_t resolve_positive_or_default(const int32_t value, const int32_t fallback) noexcept {
  const int32_t use_value = static_cast<int32_t>(value > 0);
  return use_value * value + (1 - use_value) * fallback;
}

inline int32_t blocks_for_length(const int32_t block_tokens, const int32_t token_count) noexcept {
  const int32_t positive_tokens = static_cast<int32_t>(token_count > 0);
  const int32_t positive_block_tokens = static_cast<int32_t>(block_tokens > 0);
  const int32_t safe_block_tokens = block_tokens + static_cast<int32_t>(block_tokens <= 0);
  const int32_t effective_tokens = positive_tokens * positive_block_tokens * token_count;
  const int32_t rounded = (effective_tokens + safe_block_tokens - 1) / safe_block_tokens;
  return rounded * positive_block_tokens;
}

inline void reset_runtime(context & ctx) noexcept {
  ctx.sequence_active.fill(false);
  ctx.sequence_length.fill(0);
  ctx.sequence_block_count.fill(0);
  for (auto & row : ctx.seq_to_blocks) {
    row.fill(0);
  }

  ctx.block_refs.reset();
  ctx.free_count = ctx.max_blocks;
  for (int32_t i = 0; i < ctx.max_blocks; ++i) {
    ctx.free_stack[static_cast<size_t>(i)] = static_cast<uint16_t>(i);
  }
}

inline void fill_snapshot(const context & ctx, view::snapshot & snapshot) noexcept {
  snapshot = view::snapshot{};
  snapshot.max_sequences = ctx.max_sequences;
  snapshot.block_tokens = ctx.block_tokens;
  for (int32_t seq_id = 0; seq_id < ctx.max_sequences; ++seq_id) {
    const size_t seq_index = static_cast<size_t>(seq_id);
    const int32_t active = static_cast<int32_t>(ctx.sequence_active[seq_index]);
    snapshot.sequence_active[seq_index] = static_cast<uint8_t>(active);
    snapshot.sequence_recurrent_slot[seq_index] = -1;
    snapshot.sequence_length_values[seq_index] = ctx.sequence_length[seq_index] * active;
    const int32_t block_count = ctx.sequence_block_count[seq_index] * active;
    snapshot.sequence_kv_block_count[seq_index] = block_count;
    for (size_t block = 0; block < kv::detail::max_blocks_per_sequence; ++block) {
      const int32_t in_range = static_cast<int32_t>(static_cast<int32_t>(block) < block_count);
      snapshot.sequence_kv_blocks[seq_index][block] =
          static_cast<uint16_t>(ctx.seq_to_blocks[seq_index][block] * in_range);
    }
  }
}

}  // namespace detail

struct begin_reserve {
  void operator()(const event::reserve_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.resolved_max_sequences = 0;
    ev.ctx.resolved_max_blocks = 0;
    ev.ctx.resolved_block_tokens = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_allocate_sequence {
  void operator()(const event::allocate_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
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
    ev.ctx.old_blocks = 0;
    ev.ctx.new_blocks = 0;
    ev.ctx.existing_block_count = 0;
    ev.ctx.blocks_needed = 0;
    ev.ctx.linked_count = 0;
    ev.block_count_out = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_branch_sequence {
  void operator()(const event::branch_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.parent_blocks = 0;
    ev.ctx.linked_count = 0;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_free_sequence {
  void operator()(const event::free_sequence_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.operation_error = emel::error::cast(error::none);
    ev.ctx.block_count = 0;
    ev.ctx.unlinked_count = 0;
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
    ev.ctx.existing_block_count = 0;
    ev.ctx.new_blocks = 0;
    ev.ctx.remove_count = 0;
    ev.ctx.unlinked_count = 0;
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

struct exec_reserve {
  void operator()(const event::reserve_runtime & ev, context & ctx) const noexcept {
    ev.ctx.resolved_max_sequences =
        detail::resolve_positive_or_default(ev.request.max_sequences, kv::detail::max_sequences);
    ev.ctx.resolved_max_blocks =
        detail::resolve_positive_or_default(ev.request.max_blocks, kv::detail::max_blocks);
    ev.ctx.resolved_block_tokens =
        detail::resolve_positive_or_default(ev.request.block_tokens, kv::detail::default_block_tokens);

    ctx.max_sequences = ev.ctx.resolved_max_sequences;
    ctx.max_blocks = ev.ctx.resolved_max_blocks;
    ctx.block_tokens = ev.ctx.resolved_block_tokens;
    detail::reset_runtime(ctx);
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_allocate_sequence {
  void operator()(const event::allocate_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    const int32_t was_active = static_cast<int32_t>(ctx.sequence_active[seq_index]);
    const int32_t keep_existing = was_active;

    ctx.sequence_active[seq_index] = true;
    ctx.sequence_length[seq_index] *= keep_existing;
    ctx.sequence_block_count[seq_index] *= keep_existing;
    ev.ctx.accepted = true;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_allocate_slots {
  void operator()(const event::allocate_slots_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.old_length = ctx.sequence_length[seq_index];
    ev.ctx.new_length = ev.ctx.old_length + ev.request.token_count;
    ev.ctx.existing_block_count = ctx.sequence_block_count[seq_index];
    ev.ctx.old_blocks = detail::blocks_for_length(ctx.block_tokens, ev.ctx.old_length);
    ev.ctx.new_blocks = detail::blocks_for_length(ctx.block_tokens, ev.ctx.new_length);
    ev.ctx.blocks_needed = ev.ctx.new_blocks - ev.ctx.old_blocks;
    ev.ctx.linked_count = 0;

    for (int32_t i = 0; i < ev.ctx.blocks_needed; ++i) {
      const int32_t stack_index = ctx.free_count - 1 - i;
      const uint16_t block_id = ctx.free_stack[static_cast<size_t>(stack_index)];
      ctx.seq_to_blocks[seq_index][static_cast<size_t>(ev.ctx.existing_block_count + i)] = block_id;
      ev.ctx.linked_count += static_cast<int32_t>(
          ctx.block_refs.process_indexed<kv::detail::block_link>(static_cast<size_t>(block_id)));
    }
    ctx.free_count -= ev.ctx.blocks_needed;

    const int32_t linked_all = static_cast<int32_t>(ev.ctx.linked_count == ev.ctx.blocks_needed);
    ctx.sequence_block_count[seq_index] =
        linked_all * ev.ctx.new_blocks + (1 - linked_all) * ctx.sequence_block_count[seq_index];
    ctx.sequence_length[seq_index] =
        linked_all * ev.ctx.new_length + (1 - linked_all) * ctx.sequence_length[seq_index];
    ev.ctx.block_count = linked_all * ev.ctx.blocks_needed;
    ev.ctx.accepted = linked_all != 0;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_branch_sequence {
  void operator()(const event::branch_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t parent_index = static_cast<size_t>(ev.request.parent_seq_id);
    const size_t child_index = static_cast<size_t>(ev.request.child_seq_id);
    ev.ctx.parent_blocks = ctx.sequence_block_count[parent_index];
    ev.ctx.linked_count = 0;

    for (int32_t i = 0; i < ev.ctx.parent_blocks; ++i) {
      const uint16_t block_id = ctx.seq_to_blocks[parent_index][static_cast<size_t>(i)];
      ctx.seq_to_blocks[child_index][static_cast<size_t>(i)] = block_id;
      ev.ctx.linked_count += static_cast<int32_t>(
          ctx.block_refs.process_indexed<kv::detail::block_link>(static_cast<size_t>(block_id)));
    }

    const int32_t linked_all = static_cast<int32_t>(ev.ctx.linked_count == ev.ctx.parent_blocks);
    ctx.sequence_active[child_index] = linked_all != 0;
    ctx.sequence_length[child_index] =
        linked_all * ctx.sequence_length[parent_index] +
        (1 - linked_all) * ctx.sequence_length[child_index];
    ctx.sequence_block_count[child_index] =
        linked_all * ev.ctx.parent_blocks + (1 - linked_all) * ctx.sequence_block_count[child_index];
    ev.ctx.accepted = linked_all != 0;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_free_sequence {
  void operator()(const event::free_sequence_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    const int32_t was_active = static_cast<int32_t>(ctx.sequence_active[seq_index]);
    ev.ctx.block_count = ctx.sequence_block_count[seq_index] * was_active;
    ev.ctx.unlinked_count = 0;

    for (int32_t i = 0; i < ev.ctx.block_count; ++i) {
      const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(i)];
      ev.ctx.unlinked_count += static_cast<int32_t>(
          ctx.block_refs.process_indexed<kv::detail::block_unlink>(static_cast<size_t>(block_id)));
    }

    const int32_t unlink_ok = static_cast<int32_t>(ev.ctx.unlinked_count == ev.ctx.block_count);
    const int32_t allow_recycle = was_active * unlink_ok;
    const auto & refs = ctx.block_refs.storage().refs;
    int32_t free_write = ctx.free_count;
    for (int32_t i = 0; i < ev.ctx.block_count; ++i) {
      const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(i)];
      const int32_t should_recycle =
          allow_recycle * static_cast<int32_t>(refs[static_cast<size_t>(block_id)] == 0);
      ctx.free_stack[static_cast<size_t>(free_write)] = block_id;
      free_write += should_recycle;
    }

    ctx.free_count = allow_recycle * free_write + (1 - allow_recycle) * ctx.free_count;
    const int32_t keep_sequence = 1 - allow_recycle;
    ctx.sequence_active[seq_index] = (keep_sequence != 0) && ctx.sequence_active[seq_index];
    ctx.sequence_length[seq_index] *= keep_sequence;
    ctx.sequence_block_count[seq_index] *= keep_sequence;

    const int32_t accepted = (1 - was_active) + allow_recycle;
    ev.ctx.accepted = accepted != 0;
    ev.ctx.operation_error = emel::error::cast(error::none);
  }
};

struct exec_rollback_slots {
  void operator()(const event::rollback_slots_runtime & ev, context & ctx) const noexcept {
    const size_t seq_index = static_cast<size_t>(ev.request.seq_id);
    ev.ctx.current_length = ctx.sequence_length[seq_index];
    ev.ctx.new_length = std::max(0, ev.ctx.current_length - ev.request.token_count);
    ev.ctx.existing_block_count = ctx.sequence_block_count[seq_index];
    ev.ctx.new_blocks = detail::blocks_for_length(ctx.block_tokens, ev.ctx.new_length);
    ev.ctx.remove_count = ev.ctx.existing_block_count - ev.ctx.new_blocks;
    ev.ctx.unlinked_count = 0;

    const int32_t plan_valid =
        static_cast<int32_t>(ev.ctx.current_length >= 0 && ev.ctx.new_blocks <= ev.ctx.existing_block_count);
    const int32_t effective_remove_count = plan_valid * ev.ctx.remove_count;
    for (int32_t i = 0; i < effective_remove_count; ++i) {
      const int32_t block_index = ev.ctx.existing_block_count - 1 - i;
      const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(block_index)];
      ev.ctx.unlinked_count += static_cast<int32_t>(
          ctx.block_refs.process_indexed<kv::detail::block_unlink>(static_cast<size_t>(block_id)));
    }

    const int32_t unlink_ok = static_cast<int32_t>(ev.ctx.unlinked_count == effective_remove_count);
    const int32_t apply_update = plan_valid * unlink_ok;
    const auto & refs = ctx.block_refs.storage().refs;
    int32_t free_write = ctx.free_count;
    for (int32_t i = 0; i < effective_remove_count; ++i) {
      const int32_t block_index = ev.ctx.existing_block_count - 1 - i;
      const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(block_index)];
      const int32_t should_recycle =
          apply_update * static_cast<int32_t>(refs[static_cast<size_t>(block_id)] == 0);
      ctx.free_stack[static_cast<size_t>(free_write)] = block_id;
      free_write += should_recycle;
    }

    ctx.free_count = apply_update * free_write + (1 - apply_update) * ctx.free_count;
    ctx.sequence_block_count[seq_index] =
        apply_update * ev.ctx.new_blocks + (1 - apply_update) * ctx.sequence_block_count[seq_index];
    ctx.sequence_length[seq_index] =
        apply_update * ev.ctx.new_length + (1 - apply_update) * ctx.sequence_length[seq_index];
    ev.ctx.block_count = apply_update * effective_remove_count;
    ev.ctx.accepted = apply_update != 0;
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
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_out_of_memory {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::out_of_memory);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_operation {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = kv::detail::cast_api_error(runtime_ev.ctx.operation_error);
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
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
    auto & runtime_ev = kv::detail::unwrap_runtime_event(ev);
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
inline constexpr exec_reserve exec_reserve{};
inline constexpr exec_allocate_sequence exec_allocate_sequence{};
inline constexpr exec_allocate_slots exec_allocate_slots{};
inline constexpr exec_branch_sequence exec_branch_sequence{};
inline constexpr exec_free_sequence exec_free_sequence{};
inline constexpr exec_rollback_slots exec_rollback_slots{};
inline constexpr exec_capture_view exec_capture_view{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_out_of_memory mark_out_of_memory{};
inline constexpr mark_error_from_operation mark_error_from_operation{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::memory::kv::action
