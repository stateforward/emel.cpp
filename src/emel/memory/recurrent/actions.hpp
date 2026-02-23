#pragma once

#include <algorithm>

#include "emel/memory/recurrent/context.hpp"

namespace emel::memory::recurrent::action {

inline void reset_context_state(context &ctx) noexcept {
  ctx.slot_capacity = 0;
  ctx.active_count = 0;
  ctx.seq_to_slot.fill(SLOT_NONE);
  ctx.slot_active.fill(0);
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.reserve_request = {};
  ctx.allocate_request = {};
  ctx.branch_request = {};
  ctx.free_request = {};
}

inline context::context() { reset_context_state(*this); }

inline bool has_capacity(const context &ctx) noexcept {
  return ctx.slot_capacity > 0;
}

inline int32_t slot_for_sequence(const context &ctx, int32_t seq_id) noexcept {
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    return SLOT_NONE;
  }
  return ctx.seq_to_slot[seq_id];
}

inline bool sequence_exists(const context &ctx, int32_t seq_id) noexcept {
  const int32_t slot = slot_for_sequence(ctx, seq_id);
  if (slot == SLOT_NONE) {
    return false;
  }
  if (slot < 0 || slot >= ctx.slot_capacity) {
    return false;
  }
  return ctx.slot_active[slot] != 0;
}

inline int32_t find_free_slot(const context &ctx) noexcept {
  for (int32_t slot = 0; slot < ctx.slot_capacity; ++slot) {
    if (ctx.slot_active[slot] == 0) {
      return slot;
    }
  }
  return SLOT_NONE;
}

inline void store_reserve_request(const event::reserve &ev,
                                  context &ctx) noexcept {
  ctx.reserve_request = ev;
  ctx.reserve_request.error_out = nullptr;
}

inline void store_allocate_request(const event::allocate_sequence &ev,
                                   context &ctx) noexcept {
  ctx.allocate_request = ev;
  ctx.allocate_request.error_out = nullptr;
}

inline void store_branch_request(const event::branch_sequence &ev,
                                 context &ctx) noexcept {
  ctx.branch_request = ev;
  ctx.branch_request.error_out = nullptr;
}

inline void store_free_request(const event::free_sequence &ev,
                               context &ctx) noexcept {
  ctx.free_request = ev;
  ctx.free_request.error_out = nullptr;
}

inline void clear_requests(context &ctx) noexcept {
  ctx.reserve_request = {};
  ctx.allocate_request = {};
  ctx.branch_request = {};
  ctx.free_request = {};
}

inline constexpr auto begin_reserve = [](const event::reserve &ev,
                                         context &ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_reserve_request(ev, ctx);
};

inline constexpr auto begin_allocate_sequence =
    [](const event::allocate_sequence &ev, context &ctx) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      ctx.phase_error = EMEL_OK;
      ctx.last_error = EMEL_OK;
      store_allocate_request(ev, ctx);
    };

inline constexpr auto begin_branch_sequence =
    [](const event::branch_sequence &ev, context &ctx) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      ctx.phase_error = EMEL_OK;
      ctx.last_error = EMEL_OK;
      store_branch_request(ev, ctx);
    };

inline constexpr auto begin_free_sequence = [](const event::free_sequence &ev,
                                               context &ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_free_request(ev, ctx);
};

inline constexpr auto set_invalid_argument = [](context &ctx) {
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto run_reserve_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const int32_t capacity = ctx.reserve_request.slot_capacity;
  if (capacity <= 0 || capacity > MAX_SEQ) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  ctx.slot_capacity = capacity;
  ctx.active_count = 0;
  ctx.seq_to_slot.fill(SLOT_NONE);
  ctx.slot_active.fill(0);
};

inline constexpr auto run_allocate_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const int32_t seq_id = ctx.allocate_request.seq_id;
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  if (ctx.slot_capacity <= 0) {
    *error_out = EMEL_ERR_BACKEND;
    return;
  }
  if (sequence_exists(ctx, seq_id)) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  const int32_t slot = find_free_slot(ctx);
  if (slot == SLOT_NONE) {
    *error_out = EMEL_ERR_BACKEND;
    return;
  }
  ctx.seq_to_slot[seq_id] = slot;
  ctx.slot_active[slot] = 1;
  ctx.active_count += 1;
};

inline constexpr auto run_branch_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const int32_t seq_src = ctx.branch_request.seq_id_src;
  const int32_t seq_dst = ctx.branch_request.seq_id_dst;
  if (seq_src < 0 || seq_src >= MAX_SEQ || seq_dst < 0 || seq_dst >= MAX_SEQ ||
      seq_src == seq_dst) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  if (!sequence_exists(ctx, seq_src) || sequence_exists(ctx, seq_dst)) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  const int32_t slot = find_free_slot(ctx);
  if (slot == SLOT_NONE) {
    *error_out = EMEL_ERR_BACKEND;
    return;
  }
  ctx.seq_to_slot[seq_dst] = slot;
  ctx.slot_active[slot] = 1;
  ctx.active_count += 1;
};

inline constexpr auto run_free_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const int32_t seq_id = ctx.free_request.seq_id;
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  if (!sequence_exists(ctx, seq_id)) {
    *error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  const int32_t slot = ctx.seq_to_slot[seq_id];
  if (slot < 0 || slot >= ctx.slot_capacity) {
    *error_out = EMEL_ERR_BACKEND;
    return;
  }
  ctx.seq_to_slot[seq_id] = SLOT_NONE;
  ctx.slot_active[slot] = 0;
  if (ctx.active_count > 0) {
    ctx.active_count -= 1;
  }
};

struct run_reserve_phase {
  void operator()(context &ctx) const noexcept {
    int32_t err = EMEL_OK;
    run_reserve_step(ctx, &err);
    ctx.phase_error = err;
  }
};

struct run_allocate_phase {
  void operator()(context &ctx) const noexcept {
    int32_t err = EMEL_OK;
    run_allocate_step(ctx, &err);
    ctx.phase_error = err;
  }
};

struct run_branch_phase {
  void operator()(context &ctx) const noexcept {
    int32_t err = EMEL_OK;
    run_branch_step(ctx, &err);
    ctx.phase_error = err;
  }
};

struct run_free_phase {
  void operator()(context &ctx) const noexcept {
    int32_t err = EMEL_OK;
    run_free_step(ctx, &err);
    ctx.phase_error = err;
  }
};

struct mark_done {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context &ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error =
        ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct clear_request {
  void operator()(context &ctx) const noexcept { clear_requests(ctx); }
};

struct on_unexpected {
  template <class ev>
  void operator()(const ev &event, context &ctx) const noexcept {
    if constexpr (requires { event.error_out; }) {
      if (event.error_out != nullptr) {
        *event.error_out = EMEL_ERR_BACKEND;
      }
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr run_reserve_phase run_reserve_phase{};
inline constexpr run_allocate_phase run_allocate_phase{};
inline constexpr run_branch_phase run_branch_phase{};
inline constexpr run_free_phase run_free_phase{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr clear_request clear_request{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::memory::recurrent::action
