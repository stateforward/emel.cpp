#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/telemetry/provider/events.hpp"

namespace emel::telemetry::provider::action {

struct context {
  void * queue_ctx = nullptr;
  emel::telemetry::enqueue_record_fn try_enqueue = nullptr;
  int32_t max_batch = 64;

  uint64_t sessions_started = 0;
  uint64_t records_emitted = 0;
  uint64_t records_dropped = 0;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct run_configure {
  void operator()(const event::configure & ev, context & ctx) const noexcept {
    ctx.queue_ctx = ev.queue_ctx;
    ctx.try_enqueue = ev.try_enqueue;
    ctx.max_batch = ev.max_batch;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.queue_ctx == nullptr || ctx.try_enqueue == nullptr || ctx.max_batch <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = ctx.phase_error;
    }
  }
};

struct run_start {
  void operator()(const event::start & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.queue_ctx == nullptr || ctx.try_enqueue == nullptr || ctx.max_batch <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    } else {
      ctx.sessions_started += 1;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = ctx.phase_error;
    }
  }
};

struct run_publish {
  void operator()(const event::publish & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    bool dropped = false;
    if (ctx.queue_ctx == nullptr || ctx.try_enqueue == nullptr || ctx.max_batch <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      dropped = true;
    } else {
      const bool accepted = ctx.try_enqueue(ctx.queue_ctx, ev.value);
      if (!accepted) {
        dropped = true;
        ctx.records_dropped += 1;
      } else {
        ctx.records_emitted += 1;
      }
    }
    if (ev.dropped_out != nullptr) {
      *ev.dropped_out = dropped;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = ctx.phase_error;
    }
  }
};

struct run_stop {
  void operator()(const event::stop & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct run_reset {
  void operator()(const event::reset & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & ctx) const noexcept {
    set_error(ctx, EMEL_ERR_BACKEND);
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    if constexpr (requires { ev.dropped_out; }) {
      if (ev.dropped_out != nullptr) {
        *ev.dropped_out = true;
      }
    }
  }
};

inline constexpr run_configure run_configure{};
inline constexpr run_start run_start{};
inline constexpr run_publish run_publish{};
inline constexpr run_stop run_stop{};
inline constexpr run_reset run_reset{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::telemetry::provider::action
