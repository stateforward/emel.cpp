#pragma once

#include "emel/memory/coordinator/recurrent/context.hpp"

namespace emel::memory::coordinator::recurrent::action {

namespace event = emel::memory::coordinator::event;

namespace detail {

inline int32_t normalize_prepare_error(const event::memory_status status,
                                       const int32_t err) noexcept {
  if (err != EMEL_OK) {
    return err;
  }
  switch (status) {
    case event::memory_status::success:
    case event::memory_status::no_update:
      return EMEL_OK;
    case event::memory_status::failed_prepare:
    case event::memory_status::failed_compute:
      return EMEL_ERR_BACKEND;
  }
  return EMEL_ERR_BACKEND;
}

}  // namespace detail

inline void store_update_request(const event::prepare_update & ev, context & ctx) noexcept {
  ctx.update_request = ev;
  ctx.update_request.status_out = nullptr;
  ctx.update_request.error_out = nullptr;
}

inline void store_batch_request(const event::prepare_batch & ev, context & ctx) noexcept {
  ctx.batch_request = ev;
  ctx.batch_request.status_out = nullptr;
  ctx.batch_request.error_out = nullptr;
}

inline void store_full_request(const event::prepare_full & ev, context & ctx) noexcept {
  ctx.full_request = ev;
  ctx.full_request.status_out = nullptr;
  ctx.full_request.error_out = nullptr;
}

inline void clear_requests(context & ctx) noexcept {
  ctx.active_request = request_kind::none;
  ctx.update_request = {};
  ctx.batch_request = {};
  ctx.full_request = {};
  ctx.prepared_status = event::memory_status::success;
}

inline constexpr auto begin_prepare_update = [](const event::prepare_update & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.prepared_status = event::memory_status::success;
  ctx.active_request = request_kind::update;
  store_update_request(ev, ctx);
};

inline constexpr auto begin_prepare_batch = [](const event::prepare_batch & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.prepared_status = event::memory_status::success;
  ctx.active_request = request_kind::batch;
  store_batch_request(ev, ctx);
};

inline constexpr auto begin_prepare_full = [](const event::prepare_full & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.prepared_status = event::memory_status::success;
  ctx.active_request = request_kind::full;
  store_full_request(ev, ctx);
};

inline constexpr auto set_invalid_argument = [](context & ctx) {
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto set_backend_error = [](context & ctx) {
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_ERR_BACKEND;
};

inline constexpr auto run_prepare_update_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
  ctx.has_pending_update = false;
  ctx.prepared_status = event::memory_status::no_update;
  if (ctx.update_request.prepare_fn == nullptr) {
    return;
  }
  int32_t err = EMEL_OK;
  const event::memory_status status = ctx.update_request.prepare_fn(
    ctx.update_request,
    ctx.update_request.prepare_ctx,
    &err);
  ctx.prepared_status = status;
  ctx.phase_error = detail::normalize_prepare_error(status, err);
  if (ctx.phase_error == EMEL_OK && status == event::memory_status::success) {
    ctx.has_pending_update = true;
  }
};

inline constexpr auto run_prepare_batch_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
  ctx.batch_prepare_count += 1;
  ctx.prepared_status = event::memory_status::success;
  if (ctx.batch_request.prepare_fn == nullptr) {
    return;
  }
  int32_t err = EMEL_OK;
  const event::memory_status status = ctx.batch_request.prepare_fn(
    ctx.batch_request,
    ctx.batch_request.prepare_ctx,
    &err);
  ctx.prepared_status = status;
  ctx.phase_error = detail::normalize_prepare_error(status, err);
};

inline constexpr auto run_prepare_full_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
  ctx.full_prepare_count += 1;
  ctx.prepared_status = event::memory_status::success;
  if (ctx.full_request.prepare_fn == nullptr) {
    return;
  }
  int32_t err = EMEL_OK;
  const event::memory_status status = ctx.full_request.prepare_fn(
    ctx.full_request,
    ctx.full_request.prepare_ctx,
    &err);
  ctx.prepared_status = status;
  ctx.phase_error = detail::normalize_prepare_error(status, err);
};

inline constexpr auto run_apply_update_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
  ctx.has_pending_update = false;
  ctx.update_apply_count += 1;
};

inline constexpr auto run_publish_update_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
};

inline constexpr auto run_publish_batch_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
};

inline constexpr auto run_publish_full_phase = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
};

inline constexpr auto mark_done = [](context & ctx) {
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct clear_request {
  void operator()(context & ctx) const noexcept { clear_requests(ctx); }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr ensure_last_error ensure_last_error{};
inline constexpr clear_request clear_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::memory::coordinator::recurrent::action
