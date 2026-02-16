#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"

namespace emel::memory::coordinator::action {

struct context {
  bool has_pending_update = false;
  int32_t update_apply_count = 0;
  int32_t batch_prepare_count = 0;
  int32_t full_prepare_count = 0;
};

inline constexpr auto begin_prepare_update = [](const event::prepare_update & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};
inline constexpr auto begin_prepare_batch = [](const event::prepare_batch & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};
inline constexpr auto begin_prepare_full = [](const event::prepare_full & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto run_validate_update = [](const event::validate_update & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto run_validate_batch = [](const event::validate_batch & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  if (ev.request->n_ubatch <= 0 || ev.request->n_ubatches_total <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto run_validate_full = [](const event::validate_full & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto run_prepare_update_step = [](const event::prepare_update_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr || ev.prepared_status_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  *ev.prepared_status_out = (ev.request->optimize || ctx.has_pending_update) ?
      event::memory_status::success : event::memory_status::no_update;
};

inline constexpr auto run_prepare_batch_step = [](const event::prepare_batch_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr || ev.prepared_status_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  ctx.batch_prepare_count += 1;
  ctx.has_pending_update = true;
  *ev.prepared_status_out = event::memory_status::success;
};

inline constexpr auto run_prepare_full_step = [](const event::prepare_full_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr || ev.prepared_status_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  ctx.full_prepare_count += 1;
  ctx.has_pending_update = true;
  *ev.prepared_status_out = event::memory_status::success;
};

inline constexpr auto run_apply_update_step = [](const event::apply_update_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (ev.prepared_status != event::memory_status::success) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (!ctx.has_pending_update && !ev.request->optimize) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  ctx.has_pending_update = false;
  ctx.update_apply_count += 1;
};

inline constexpr auto run_publish_update = [](const event::publish_update & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request != nullptr && ev.request->status_out != nullptr) {
    *ev.request->status_out = ev.prepared_status;
  }
};

inline constexpr auto run_publish_batch = [](const event::publish_batch & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request != nullptr && ev.request->status_out != nullptr) {
    *ev.request->status_out = ev.prepared_status;
  }
};

inline constexpr auto run_publish_full = [](const event::publish_full & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.request != nullptr && ev.request->status_out != nullptr) {
    *ev.request->status_out = ev.prepared_status;
  }
};

inline constexpr auto on_memory_done = [](const events::memory_done &, context &) {};
inline constexpr auto on_memory_error = [](const events::memory_error &, context &) {};

}  // namespace emel::memory::coordinator::action
