#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/telemetry/provider/events.hpp"

namespace emel::telemetry::provider::action {

struct context {
  void * queue_ctx = nullptr;
  emel::telemetry::enqueue_record_fn try_enqueue = nullptr;
  int32_t max_batch = 64;

  emel::telemetry::record pending_record = {};
  bool pending_dropped = false;

  uint64_t sessions_started = 0;
  uint64_t records_emitted = 0;
  uint64_t records_dropped = 0;
};

inline constexpr auto begin_configure = [](const event::configure & ev, context & ctx) {
  ctx.queue_ctx = ev.queue_ctx;
  ctx.try_enqueue = ev.try_enqueue;
  ctx.max_batch = ev.max_batch;
};

inline constexpr auto run_validate_config = [](const event::validate_config & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.queue_ctx == nullptr || ctx.try_enqueue == nullptr || ctx.max_batch <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto begin_start = [](const event::start &, context & ctx) {
  ctx.pending_dropped = false;
};

inline constexpr auto run_start = [](const event::run_start & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.queue_ctx == nullptr || ctx.try_enqueue == nullptr || ctx.max_batch <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  ctx.sessions_started += 1;
};

inline constexpr auto begin_publish = [](const event::publish & ev, context & ctx) {
  ctx.pending_record = ev.value;
  ctx.pending_dropped = false;
};

inline constexpr auto run_publish_record = [](const event::publish_record & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.try_enqueue == nullptr || ctx.queue_ctx == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  const bool accepted = ctx.try_enqueue(ctx.queue_ctx, ctx.pending_record);
  if (!accepted) {
    ctx.pending_dropped = true;
    ctx.records_dropped += 1;
    return;
  }

  ctx.pending_dropped = false;
  ctx.records_emitted += 1;
};

inline constexpr auto begin_stop = [](const event::stop &, context & ctx) {
  ctx.pending_dropped = false;
};

inline constexpr auto run_stop = [](const event::run_stop & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_reset = [](const event::reset & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.pending_dropped = false;
};

}  // namespace emel::telemetry::provider::action

namespace emel::telemetry::provider::guard {

inline bool is_configured(const action::context & ctx) {
  return ctx.queue_ctx != nullptr && ctx.try_enqueue != nullptr && ctx.max_batch > 0;
}

}  // namespace emel::telemetry::provider::guard

