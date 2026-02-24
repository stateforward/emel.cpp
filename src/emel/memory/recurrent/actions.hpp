#pragma once

#include <algorithm>

#include "emel/memory/recurrent/context.hpp"
#include "emel/memory/recurrent/events.hpp"

namespace emel::memory::recurrent::action {

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

inline bool valid_sequence_id(const context & ctx, const int32_t seq_id) noexcept {
  return seq_id >= 0 && seq_id < ctx.max_sequences;
}

inline bool is_active(const context & ctx, const int32_t seq_id) noexcept {
  return valid_sequence_id(ctx, seq_id) &&
         ctx.seq_to_slot[static_cast<size_t>(seq_id)] != INVALID_SLOT;
}

inline int32_t lookup_slot(const context & ctx, const int32_t seq_id) noexcept {
  if (!is_active(ctx, seq_id)) {
    return INVALID_SLOT;
  }
  return ctx.seq_to_slot[static_cast<size_t>(seq_id)];
}

inline int32_t sequence_length_value(const context & ctx, const int32_t seq_id) noexcept {
  if (!is_active(ctx, seq_id)) {
    return 0;
  }
  return ctx.sequence_length[static_cast<size_t>(seq_id)];
}

inline void reset_runtime(context & ctx) noexcept {
  ctx.slots.reset();
  ctx.seq_to_slot.fill(INVALID_SLOT);
  ctx.slot_owner_seq.fill(INVALID_SLOT);
  ctx.sequence_length.fill(0);
}

inline int32_t find_first_free_slot(const context & ctx) noexcept {
  const auto & active = ctx.slots.storage().active;
  for (int32_t i = 0; i < ctx.max_slots; ++i) {
    if (active[static_cast<size_t>(i)] == 0) {
      return i;
    }
  }
  return INVALID_SLOT;
}

inline bool activate_slot(context & ctx, const int32_t slot_id) noexcept {
  if (slot_id < 0 || slot_id >= ctx.max_slots) {
    return false;
  }
  return ctx.slots.process_indexed<slot_activate>(static_cast<size_t>(slot_id));
}

inline bool deactivate_slot(context & ctx, const int32_t slot_id) noexcept {
  if (slot_id < 0 || slot_id >= ctx.max_slots) {
    return false;
  }
  return ctx.slots.process_indexed<slot_deactivate>(static_cast<size_t>(slot_id));
}

inline void run_reserve_phase(const event::reserve & ev, context & ctx) noexcept {
  begin_phase(ctx);

  const int32_t max_sequences = ev.max_sequences > 0 ? ev.max_sequences : MAX_SEQUENCES;
  const int32_t requested_slots = ev.max_blocks > 0 ? ev.max_blocks : max_sequences;
  if (max_sequences <= 0 || max_sequences > MAX_SEQUENCES || requested_slots <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.max_sequences = max_sequences;
  ctx.max_slots = std::min(max_sequences, requested_slots);
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

  if (is_active(ctx, ev.seq_id)) {
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t slot_id = find_first_free_slot(ctx);
  if (slot_id == INVALID_SLOT) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.phase_out_of_memory = true;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (!activate_slot(ctx, slot_id)) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.seq_to_slot[static_cast<size_t>(ev.seq_id)] = slot_id;
  ctx.slot_owner_seq[static_cast<size_t>(slot_id)] = ev.seq_id;
  ctx.sequence_length[static_cast<size_t>(ev.seq_id)] = 0;

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_allocate_slots_phase(const event::allocate_slots & ev, context & ctx) noexcept {
  begin_phase(ctx);

  if (!is_active(ctx, ev.seq_id) || ev.token_count <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.sequence_length[static_cast<size_t>(ev.seq_id)] += ev.token_count;
  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = 0;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_branch_sequence_phase(const event::branch_sequence & ev,
                                      context & ctx) noexcept {
  begin_phase(ctx);

  if (!is_active(ctx, ev.parent_seq_id) || !valid_sequence_id(ctx, ev.child_seq_id) ||
      ev.parent_seq_id == ev.child_seq_id || is_active(ctx, ev.child_seq_id)) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (ev.copy_state == nullptr) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t child_slot = find_first_free_slot(ctx);
  if (child_slot == INVALID_SLOT) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.phase_out_of_memory = true;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  if (!activate_slot(ctx, child_slot)) {
    ctx.phase_error = EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const int32_t src_slot = lookup_slot(ctx, ev.parent_seq_id);
  int32_t copy_error = EMEL_OK;
  const bool copied = ev.copy_state(src_slot, child_slot, ev.copy_state_user_data, &copy_error);
  if (!copied || copy_error != EMEL_OK) {
    (void)deactivate_slot(ctx, child_slot);
    ctx.phase_error = copy_error != EMEL_OK ? copy_error : EMEL_ERR_BACKEND;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  ctx.seq_to_slot[static_cast<size_t>(ev.child_seq_id)] = child_slot;
  ctx.slot_owner_seq[static_cast<size_t>(child_slot)] = ev.child_seq_id;
  ctx.sequence_length[static_cast<size_t>(ev.child_seq_id)] =
      ctx.sequence_length[static_cast<size_t>(ev.parent_seq_id)];

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

  const int32_t slot_id = lookup_slot(ctx, ev.seq_id);
  if (slot_id != INVALID_SLOT) {
    if (!deactivate_slot(ctx, slot_id)) {
      ctx.phase_error = EMEL_ERR_BACKEND;
      end_phase(ctx);
      write_error(ctx.phase_error, ev.error_out);
      return;
    }

    ctx.slot_owner_seq[static_cast<size_t>(slot_id)] = INVALID_SLOT;
    ctx.seq_to_slot[static_cast<size_t>(ev.seq_id)] = INVALID_SLOT;
    ctx.sequence_length[static_cast<size_t>(ev.seq_id)] = 0;
  }

  end_phase(ctx);
  write_error(ctx.phase_error, ev.error_out);
}

inline void run_rollback_slots_phase(const event::rollback_slots & ev,
                                     context & ctx) noexcept {
  begin_phase(ctx);

  if (!is_active(ctx, ev.seq_id) || ev.token_count <= 0) {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    end_phase(ctx);
    write_error(ctx.phase_error, ev.error_out);
    return;
  }

  const size_t index = static_cast<size_t>(ev.seq_id);
  ctx.sequence_length[index] = std::max(0, ctx.sequence_length[index] - ev.token_count);
  if (ev.block_count_out != nullptr) {
    *ev.block_count_out = 0;
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

}  // namespace emel::memory::recurrent::action
