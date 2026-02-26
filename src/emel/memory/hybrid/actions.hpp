#pragma once

#include "emel/memory/hybrid/context.hpp"
#include "emel/memory/hybrid/events.hpp"

namespace emel::memory::hybrid::action {

inline int32_t normalize_error(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err != EMEL_OK) {
    return err;
  }
  return EMEL_ERR_BACKEND;
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

inline void run_reserve_phase(const event::reserve & ev, context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_error = EMEL_OK;
  if (!ctx.kv.process_event(event::reserve{
        .max_sequences = ev.max_sequences,
        .max_blocks = ev.max_blocks,
        .block_tokens = ev.block_tokens,
        .error_out = &kv_error,
      })) {
    ctx.phase_error = normalize_error(false, kv_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  if (!ctx.recurrent.process_event(event::reserve{
        .max_sequences = ev.max_sequences,
        .max_blocks = ev.max_blocks,
        .block_tokens = ev.block_tokens,
        .error_out = &recurrent_error,
      })) {
    ctx.phase_error = normalize_error(false, recurrent_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_allocate_sequence_phase(const event::allocate_sequence & ev,
                                        context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_error = EMEL_OK;
  const bool kv_ok = ctx.kv.process_event(event::allocate_sequence{
    .seq_id = ev.seq_id,
    .error_out = &kv_error,
  });
  if (!kv_ok) {
    ctx.phase_error = normalize_error(kv_ok, kv_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent.process_event(event::allocate_sequence{
    .seq_id = ev.seq_id,
    .error_out = &recurrent_error,
  });
  if (!recurrent_ok) {
    int32_t rollback_error = EMEL_OK;
    (void)ctx.kv.process_event(event::free_sequence{
      .seq_id = ev.seq_id,
      .error_out = &rollback_error,
    });
    ctx.phase_error = normalize_error(recurrent_ok, recurrent_error);
    ctx.phase_out_of_memory = ctx.phase_error == EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_allocate_slots_phase(const event::allocate_slots & ev, context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_blocks = 0;
  int32_t kv_error = EMEL_OK;
  const bool kv_ok = ctx.kv.process_event(event::allocate_slots{
    .seq_id = ev.seq_id,
    .token_count = ev.token_count,
    .block_count_out = &kv_blocks,
    .error_out = &kv_error,
  });
  if (!kv_ok) {
    ctx.phase_error = normalize_error(kv_ok, kv_error);
    ctx.phase_out_of_memory = ctx.phase_error == EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent.process_event(event::allocate_slots{
    .seq_id = ev.seq_id,
    .token_count = ev.token_count,
    .error_out = &recurrent_error,
  });
  if (!recurrent_ok) {
    int32_t rollback_error = EMEL_OK;
    (void)ctx.kv.process_event(event::rollback_slots{
      .seq_id = ev.seq_id,
      .token_count = ev.token_count,
      .error_out = &rollback_error,
    });
    ctx.phase_error = normalize_error(recurrent_ok, recurrent_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = kv_blocks;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_branch_sequence_phase(const event::branch_sequence & ev,
                                      context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_error = EMEL_OK;
  const bool kv_ok = ctx.kv.process_event(event::branch_sequence{
    .parent_seq_id = ev.parent_seq_id,
    .child_seq_id = ev.child_seq_id,
    .error_out = &kv_error,
  });
  if (!kv_ok) {
    ctx.phase_error = normalize_error(kv_ok, kv_error);
    ctx.phase_out_of_memory = ctx.phase_error == EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent.process_event(event::branch_sequence{
    .parent_seq_id = ev.parent_seq_id,
    .child_seq_id = ev.child_seq_id,
    .copy_state = ev.copy_state,
    .copy_state_user_data = ev.copy_state_user_data,
    .error_out = &recurrent_error,
  });
  if (!recurrent_ok) {
    int32_t rollback_error = EMEL_OK;
    (void)ctx.kv.process_event(event::free_sequence{
      .seq_id = ev.child_seq_id,
      .error_out = &rollback_error,
    });
    ctx.phase_error = normalize_error(recurrent_ok, recurrent_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_free_sequence_phase(const event::free_sequence & ev, context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_error = EMEL_OK;
  const bool kv_ok = ctx.kv.process_event(event::free_sequence{
    .seq_id = ev.seq_id,
    .error_out = &kv_error,
  });
  if (!kv_ok) {
    ctx.phase_error = normalize_error(kv_ok, kv_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent.process_event(event::free_sequence{
    .seq_id = ev.seq_id,
    .error_out = &recurrent_error,
  });
  if (!recurrent_ok) {
    ctx.phase_error = normalize_error(recurrent_ok, recurrent_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_rollback_slots_phase(const event::rollback_slots & ev,
                                     context & ctx) noexcept {
  begin_phase(ctx);

  int32_t kv_blocks = 0;
  int32_t kv_error = EMEL_OK;
  const bool kv_ok = ctx.kv.process_event(event::rollback_slots{
    .seq_id = ev.seq_id,
    .token_count = ev.token_count,
    .block_count_out = &kv_blocks,
    .error_out = &kv_error,
  });
  if (!kv_ok) {
    ctx.phase_error = normalize_error(kv_ok, kv_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  int32_t recurrent_error = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent.process_event(event::rollback_slots{
    .seq_id = ev.seq_id,
    .token_count = ev.token_count,
    .error_out = &recurrent_error,
  });
  if (!recurrent_ok) {
    ctx.phase_error = normalize_error(recurrent_ok, recurrent_error);
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = kv_blocks;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void merge_snapshots(const view::snapshot & kv_snapshot,
                            const view::snapshot & recurrent_snapshot,
                            view::snapshot & out) noexcept {
  out = view::snapshot{};
  out.max_sequences = std::min(kv_snapshot.max_sequences, recurrent_snapshot.max_sequences);
  out.block_tokens = kv_snapshot.block_tokens;
  for (int32_t seq_id = 0; seq_id < out.max_sequences; ++seq_id) {
    const size_t seq = static_cast<size_t>(seq_id);
    const bool active = kv_snapshot.sequence_active[seq] != 0 &&
                        recurrent_snapshot.sequence_active[seq] != 0;
    out.sequence_active[seq] = active ? 1u : 0u;
    if (!active) {
      continue;
    }
    out.sequence_length_values[seq] =
        std::min(kv_snapshot.sequence_length_values[seq],
                 recurrent_snapshot.sequence_length_values[seq]);
    out.sequence_kv_block_count[seq] = kv_snapshot.sequence_kv_block_count[seq];
    out.sequence_kv_blocks[seq] = kv_snapshot.sequence_kv_blocks[seq];
    out.sequence_recurrent_slot[seq] = recurrent_snapshot.sequence_recurrent_slot[seq];
  }
}

struct capture_view {
  void operator()(const event::capture_view & ev, context & ctx) const noexcept {
    if (ev.snapshot_out == nullptr || ctx.kv_snapshot == nullptr ||
        ctx.recurrent_snapshot == nullptr) {
      write_error(EMEL_ERR_INVALID_ARGUMENT, ev.error_out);
      return;
    }

    int32_t kv_error = EMEL_OK;
    (void)ctx.kv.process_event(event::capture_view{
      .snapshot_out = ctx.kv_snapshot.get(),
      .error_out = &kv_error,
    });
    if (kv_error != EMEL_OK) {
      write_error(kv_error, ev.error_out);
      return;
    }

    int32_t recurrent_error = EMEL_OK;
    (void)ctx.recurrent.process_event(event::capture_view{
      .snapshot_out = ctx.recurrent_snapshot.get(),
      .error_out = &recurrent_error,
    });
    if (recurrent_error != EMEL_OK) {
      write_error(recurrent_error, ev.error_out);
      return;
    }

    merge_snapshots(*ctx.kv_snapshot, *ctx.recurrent_snapshot, *ev.snapshot_out);
    write_error(EMEL_OK, ev.error_out);
  }
};

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

}  // namespace emel::memory::hybrid::action
