#pragma once

#include "emel/memory/hybrid/context.hpp"

namespace emel::memory::hybrid::action {

inline int32_t normalize_child_error(bool accepted, int32_t err) noexcept {
  if (err != EMEL_OK) {
    return err;
  }
  return accepted ? EMEL_OK : EMEL_ERR_BACKEND;
}

inline bool has_capacity(const context &ctx) noexcept { return ctx.reserved; }

inline bool has_sequence(const context &ctx, int32_t seq_id) noexcept {
  return ctx.kv_memory.has_sequence(seq_id) ||
         ctx.recurrent_memory.has_sequence(seq_id);
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

  const event::reserve &request = ctx.reserve_request;
  int32_t recurrent_err = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent_memory.process_event(
      emel::memory::recurrent::event::reserve{
          .slot_capacity = request.recurrent_slot_capacity,
          .error_out = &recurrent_err,
      });
  const int32_t recurrent_status =
      normalize_child_error(recurrent_ok, recurrent_err);
  if (recurrent_status != EMEL_OK) {
    *error_out = recurrent_status;
    ctx.reserved = false;
    return;
  }

  int32_t kv_err = EMEL_OK;
  const bool kv_ok =
      ctx.kv_memory.process_event(emel::memory::kv::event::reserve{
          .kv_size = request.kv_size,
          .n_stream = request.n_stream,
          .n_pad = request.n_pad,
          .n_swa = request.n_swa,
          .swa_type = request.swa_type,
          .seq_to_stream = request.seq_to_stream,
          .seq_to_stream_count = request.seq_to_stream_count,
          .error_out = &kv_err,
      });
  const int32_t kv_status = normalize_child_error(kv_ok, kv_err);
  if (kv_status != EMEL_OK) {
    *error_out = kv_status;
    ctx.reserved = false;
    return;
  }

  ctx.reserved = true;
};

inline constexpr auto run_allocate_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const event::allocate_sequence &request = ctx.allocate_request;
  int32_t kv_err = EMEL_OK;
  const bool kv_ok =
      ctx.kv_memory.process_event(emel::memory::kv::event::allocate_sequence{
          .seq_id = request.seq_id,
          .slot_count = request.slot_count,
          .error_out = &kv_err,
      });
  const int32_t kv_status = normalize_child_error(kv_ok, kv_err);
  if (kv_status != EMEL_OK) {
    *error_out = kv_status;
    return;
  }

  int32_t recurrent_err = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent_memory.process_event(
      emel::memory::recurrent::event::allocate_sequence{
          .seq_id = request.seq_id,
          .error_out = &recurrent_err,
      });
  const int32_t recurrent_status =
      normalize_child_error(recurrent_ok, recurrent_err);
  if (recurrent_status != EMEL_OK) {
    int32_t rollback_err = EMEL_OK;
    (void)ctx.kv_memory.process_event(emel::memory::kv::event::free_sequence{
        .seq_id = request.seq_id,
        .error_out = &rollback_err,
    });
    *error_out = recurrent_status;
    return;
  }
};

inline constexpr auto run_branch_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const event::branch_sequence &request = ctx.branch_request;
  int32_t kv_err = EMEL_OK;
  const bool kv_ok =
      ctx.kv_memory.process_event(emel::memory::kv::event::branch_sequence{
          .seq_id_src = request.seq_id_src,
          .seq_id_dst = request.seq_id_dst,
          .error_out = &kv_err,
      });
  const int32_t kv_status = normalize_child_error(kv_ok, kv_err);
  if (kv_status != EMEL_OK) {
    *error_out = kv_status;
    return;
  }

  int32_t recurrent_err = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent_memory.process_event(
      emel::memory::recurrent::event::branch_sequence{
          .seq_id_src = request.seq_id_src,
          .seq_id_dst = request.seq_id_dst,
          .error_out = &recurrent_err,
      });
  const int32_t recurrent_status =
      normalize_child_error(recurrent_ok, recurrent_err);
  if (recurrent_status != EMEL_OK) {
    int32_t rollback_err = EMEL_OK;
    (void)ctx.kv_memory.process_event(emel::memory::kv::event::free_sequence{
        .seq_id = request.seq_id_dst,
        .error_out = &rollback_err,
    });
    *error_out = recurrent_status;
    return;
  }
};

inline constexpr auto run_free_step = [](context &ctx, int32_t *error_out) {
  if (error_out == nullptr) {
    return;
  }
  *error_out = EMEL_OK;

  const event::free_sequence &request = ctx.free_request;
  int32_t kv_err = EMEL_OK;
  const bool kv_ok =
      ctx.kv_memory.process_event(emel::memory::kv::event::free_sequence{
          .seq_id = request.seq_id,
          .error_out = &kv_err,
      });
  int32_t status = normalize_child_error(kv_ok, kv_err);

  int32_t recurrent_err = EMEL_OK;
  const bool recurrent_ok = ctx.recurrent_memory.process_event(
      emel::memory::recurrent::event::free_sequence{
          .seq_id = request.seq_id,
          .error_out = &recurrent_err,
      });
  const int32_t recurrent_status =
      normalize_child_error(recurrent_ok, recurrent_err);

  if (status == EMEL_OK) {
    status = recurrent_status;
  }
  if (status != EMEL_OK) {
    *error_out = status;
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

} // namespace emel::memory::hybrid::action
