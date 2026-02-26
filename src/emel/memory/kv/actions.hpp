#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "emel/memory/kv/context.hpp"
#include "emel/memory/kv/events.hpp"

namespace emel::memory::kv::action {

inline bool valid_sequence_id(const context & ctx, const int32_t seq_id) noexcept {
  return seq_id >= 0 && seq_id < ctx.max_sequences;
}

inline int32_t required_blocks(const context & ctx, const int32_t token_count) noexcept {
  if (token_count <= 0 || ctx.block_tokens <= 0) {
    return 0;
  }
  return (token_count + ctx.block_tokens - 1) / ctx.block_tokens;
}

inline int32_t blocks_for_length(const context & ctx, const int32_t token_count) noexcept {
  if (token_count <= 0 || ctx.block_tokens <= 0) {
    return 0;
  }
  return (token_count + ctx.block_tokens - 1) / ctx.block_tokens;
}

inline void write_error(const int32_t err, int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = err;
  }
}

inline void begin_phase(context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  ctx.phase_out_of_memory = false;
}

inline void end_phase(context & ctx) noexcept {
  ctx.last_error = ctx.phase_error;
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
    const size_t seq = static_cast<size_t>(seq_id);
    const bool active = ctx.sequence_active[seq];
    snapshot.sequence_active[seq] = active ? 1u : 0u;
    snapshot.sequence_recurrent_slot[seq] = -1;
    if (!active) {
      continue;
    }
    snapshot.sequence_length_values[seq] = ctx.sequence_length[seq];
    const int32_t block_count = ctx.sequence_block_count[seq];
    snapshot.sequence_kv_block_count[seq] = block_count;
    for (int32_t i = 0; i < block_count; ++i) {
      snapshot.sequence_kv_blocks[seq][static_cast<size_t>(i)] =
          ctx.seq_to_blocks[seq][static_cast<size_t>(i)];
    }
  }
}

struct capture_view {
  void operator()(const event::capture_view & ev, context & ctx) const noexcept {
    if (ev.snapshot_out == nullptr) {
      write_error(EMEL_ERR_INVALID_ARGUMENT, ev.error_out);
      return;
    }
    fill_snapshot(ctx, *ev.snapshot_out);
    write_error(EMEL_OK, ev.error_out);
  }
};

inline int32_t sequence_length_value(const context & ctx, const int32_t seq_id) noexcept {
  if (!valid_sequence_id(ctx, seq_id) || !ctx.sequence_active[static_cast<size_t>(seq_id)]) {
    return 0;
  }
  return ctx.sequence_length[static_cast<size_t>(seq_id)];
}

inline int32_t lookup_block_at_pos(const context & ctx, const int32_t seq_id,
                                   const int32_t pos) noexcept {
  if (!valid_sequence_id(ctx, seq_id) || !ctx.sequence_active[static_cast<size_t>(seq_id)] ||
      pos < 0 || ctx.block_tokens <= 0) {
    return INVALID_INDEX;
  }

  const int32_t length = ctx.sequence_length[static_cast<size_t>(seq_id)];
  if (pos >= length) {
    return INVALID_INDEX;
  }

  const int32_t logical_block = pos / ctx.block_tokens;
  const int32_t block_count = ctx.sequence_block_count[static_cast<size_t>(seq_id)];
  if (logical_block < 0 || logical_block >= block_count) {
    return INVALID_INDEX;
  }

  return static_cast<int32_t>(
      ctx.seq_to_blocks[static_cast<size_t>(seq_id)][static_cast<size_t>(logical_block)]);
}

inline void run_reserve_phase(const event::reserve & ev, context & ctx) noexcept {
  begin_phase(ctx);

  const int32_t max_sequences = ev.max_sequences > 0 ? ev.max_sequences : MAX_SEQUENCES;
  const int32_t max_blocks = ev.max_blocks > 0 ? ev.max_blocks : MAX_BLOCKS;
  const int32_t block_tokens = ev.block_tokens > 0 ? ev.block_tokens : DEFAULT_BLOCK_TOKENS;

  if (max_sequences <= 0 || max_sequences > MAX_SEQUENCES || max_blocks <= 0 ||
      max_blocks > MAX_BLOCKS || block_tokens <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.max_sequences = max_sequences;
  ctx.max_blocks = max_blocks;
  ctx.block_tokens = block_tokens;
  reset_runtime(ctx);

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_allocate_sequence_phase(const event::allocate_sequence & ev,
                                        context & ctx) noexcept {
  begin_phase(ctx);

  if (!valid_sequence_id(ctx, ev.seq_id)) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t index = static_cast<size_t>(ev.seq_id);
  if (!ctx.sequence_active[index]) {
    ctx.sequence_active[index] = true;
    ctx.sequence_length[index] = 0;
    ctx.sequence_block_count[index] = 0;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_allocate_slots_phase(const event::allocate_slots & ev, context & ctx) noexcept {
  begin_phase(ctx);

  if (!valid_sequence_id(ctx, ev.seq_id) || ev.token_count <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t seq_index = static_cast<size_t>(ev.seq_id);
  if (!ctx.sequence_active[seq_index]) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }
  if (ctx.block_tokens <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t old_length = ctx.sequence_length[seq_index];
  const int64_t new_length_wide = static_cast<int64_t>(old_length) + ev.token_count;
  if (new_length_wide <= 0 || new_length_wide > std::numeric_limits<int32_t>::max()) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }
  const int32_t new_length = static_cast<int32_t>(new_length_wide);

  const int32_t block_count = ctx.sequence_block_count[seq_index];
  const int32_t old_blocks = blocks_for_length(ctx, old_length);
  const int32_t new_blocks = blocks_for_length(ctx, new_length);
  if (block_count < old_blocks || new_blocks < old_blocks) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }
  const int32_t blocks_needed = new_blocks - old_blocks;
  if (block_count + blocks_needed > MAX_BLOCKS_PER_SEQUENCE || ctx.free_count < blocks_needed) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.phase_out_of_memory = true;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (blocks_needed > 0) {
    std::array<size_t, MAX_BLOCKS_PER_SEQUENCE> ids = {};
    for (int32_t i = 0; i < blocks_needed; ++i) {
      const uint16_t block_id = ctx.free_stack[static_cast<size_t>(ctx.free_count - 1 - i)];
      ids[static_cast<size_t>(i)] = static_cast<size_t>(block_id);
      ctx.seq_to_blocks[seq_index][static_cast<size_t>(block_count + i)] = block_id;
    }
    ctx.free_count -= blocks_needed;

    const size_t linked =
        ctx.block_refs.process_indexed_batch<block_link>(ids.begin(), ids.begin() + blocks_needed);
    if (linked != static_cast<size_t>(blocks_needed)) {
      ctx.phase_error = EMEL_ERR_BACKEND;
      end_phase(ctx);
      write_error(ctx.phase_error, ev.error_out);
      return;
    }
  }

  ctx.sequence_block_count[seq_index] = new_blocks;
  ctx.sequence_length[seq_index] = new_length;

  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = blocks_needed;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_branch_sequence_phase(const event::branch_sequence & ev,
                                      context & ctx) noexcept {
  begin_phase(ctx);

  if (!valid_sequence_id(ctx, ev.parent_seq_id) || !valid_sequence_id(ctx, ev.child_seq_id) ||
      ev.parent_seq_id == ev.child_seq_id) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t parent = static_cast<size_t>(ev.parent_seq_id);
  const size_t child = static_cast<size_t>(ev.child_seq_id);

  if (!ctx.sequence_active[parent] || ctx.sequence_active[child]) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t parent_blocks = ctx.sequence_block_count[parent];
  std::array<size_t, MAX_BLOCKS_PER_SEQUENCE> ids = {};
  for (int32_t i = 0; i < parent_blocks; ++i) {
    const uint16_t block_id = ctx.seq_to_blocks[parent][static_cast<size_t>(i)];
    ctx.seq_to_blocks[child][static_cast<size_t>(i)] = block_id;
    ids[static_cast<size_t>(i)] = static_cast<size_t>(block_id);
  }

  const size_t linked =
      ctx.block_refs.process_indexed_batch<block_link>(ids.begin(), ids.begin() + parent_blocks);
  if (linked != static_cast<size_t>(parent_blocks)) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.sequence_active[child] = true;
  ctx.sequence_length[child] = ctx.sequence_length[parent];
  ctx.sequence_block_count[child] = parent_blocks;

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_free_sequence_phase(const event::free_sequence & ev, context & ctx) noexcept {
  begin_phase(ctx);

  if (!valid_sequence_id(ctx, ev.seq_id)) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t seq_index = static_cast<size_t>(ev.seq_id);
  if (!ctx.sequence_active[seq_index]) {
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t block_count = ctx.sequence_block_count[seq_index];
  std::array<size_t, MAX_BLOCKS_PER_SEQUENCE> ids = {};
  for (int32_t i = 0; i < block_count; ++i) {
    ids[static_cast<size_t>(i)] =
        static_cast<size_t>(ctx.seq_to_blocks[seq_index][static_cast<size_t>(i)]);
  }

  const size_t unlinked =
      ctx.block_refs.process_indexed_batch<block_unlink>(ids.begin(), ids.begin() + block_count);
  if (unlinked != static_cast<size_t>(block_count)) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const auto & refs = ctx.block_refs.storage().refs;
  for (int32_t i = 0; i < block_count; ++i) {
    const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(i)];
    if (refs[static_cast<size_t>(block_id)] == 0) {
      ctx.free_stack[static_cast<size_t>(ctx.free_count)] = block_id;
      ++ctx.free_count;
    }
  }

  ctx.sequence_active[seq_index] = false;
  ctx.sequence_length[seq_index] = 0;
  ctx.sequence_block_count[seq_index] = 0;

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_rollback_slots_phase(const event::rollback_slots & ev,
                                     context & ctx) noexcept {
  begin_phase(ctx);

  if (!valid_sequence_id(ctx, ev.seq_id) || ev.token_count <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t seq_index = static_cast<size_t>(ev.seq_id);
  if (!ctx.sequence_active[seq_index]) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }
  if (ctx.block_tokens <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t current_length = ctx.sequence_length[seq_index];
  if (current_length < 0) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t block_count = ctx.sequence_block_count[seq_index];
  const int32_t new_length = std::max(0, current_length - ev.token_count);
  const int32_t new_blocks = blocks_for_length(ctx, new_length);
  if (new_blocks > block_count) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }
  const int32_t remove_count = block_count - new_blocks;

  if (remove_count > 0) {
    std::array<size_t, MAX_BLOCKS_PER_SEQUENCE> ids = {};
    for (int32_t i = 0; i < remove_count; ++i) {
      const int32_t block_index = block_count - 1 - i;
      ids[static_cast<size_t>(i)] =
          static_cast<size_t>(ctx.seq_to_blocks[seq_index][static_cast<size_t>(block_index)]);
    }

    const size_t unlinked =
        ctx.block_refs.process_indexed_batch<block_unlink>(ids.begin(), ids.begin() + remove_count);
    if (unlinked != static_cast<size_t>(remove_count)) {
      ctx.phase_error = EMEL_ERR_BACKEND;
      end_phase(ctx);
      write_error(ctx.phase_error, ev.error_out);
      return;
    }

    const auto & refs = ctx.block_refs.storage().refs;
    for (int32_t i = 0; i < remove_count; ++i) {
      const int32_t block_index = block_count - 1 - i;
      const uint16_t block_id = ctx.seq_to_blocks[seq_index][static_cast<size_t>(block_index)];
      if (refs[static_cast<size_t>(block_id)] == 0) {
        ctx.free_stack[static_cast<size_t>(ctx.free_count)] = block_id;
        ++ctx.free_count;
      }
    }
  }

  ctx.sequence_block_count[seq_index] = new_blocks;
  ctx.sequence_length[seq_index] = new_length;

  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = remove_count;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

struct begin_reserve {
  void operator()(const event::reserve & ev, context & ctx) const noexcept {
    run_reserve_phase(ev, ctx);
  }
};

struct begin_allocate_sequence {
  void operator()(const event::allocate_sequence & ev, context & ctx) const noexcept {
    run_allocate_sequence_phase(ev, ctx);
  }
};

struct begin_allocate_slots {
  void operator()(const event::allocate_slots & ev, context & ctx) const noexcept {
    run_allocate_slots_phase(ev, ctx);
  }
};

struct begin_branch_sequence {
  void operator()(const event::branch_sequence & ev, context & ctx) const noexcept {
    run_branch_sequence_phase(ev, ctx);
  }
};

struct begin_free_sequence {
  void operator()(const event::free_sequence & ev, context & ctx) const noexcept {
    run_free_sequence_phase(ev, ctx);
  }
};

struct begin_rollback_slots {
  void operator()(const event::rollback_slots & ev, context & ctx) const noexcept {
    run_rollback_slots_phase(ev, ctx);
  }
};

struct clear_out_of_memory {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.phase_out_of_memory = false;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error == EMEL_OK) {
      ctx.last_error = EMEL_ERR_BACKEND;
    }
    ctx.phase_error = ctx.last_error;
    ctx.phase_out_of_memory = false;
  }
};

struct on_unexpected {
  template <class ev>
  void operator()(const ev & event, context & ctx) const noexcept {
    if constexpr (requires { event.error_out; }) {
      write_error(EMEL_ERR_BACKEND, event.error_out);
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.phase_out_of_memory = false;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr begin_reserve begin_reserve{};
inline constexpr begin_allocate_sequence begin_allocate_sequence{};
inline constexpr begin_allocate_slots begin_allocate_slots{};
inline constexpr begin_branch_sequence begin_branch_sequence{};
inline constexpr begin_free_sequence begin_free_sequence{};
inline constexpr begin_rollback_slots begin_rollback_slots{};
inline constexpr clear_out_of_memory clear_out_of_memory{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::memory::kv::action
