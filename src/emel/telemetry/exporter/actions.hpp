#pragma once

#include "emel/telemetry/exporter/context.hpp"

namespace emel::telemetry::exporter::action {

inline void clear_error_out(context & ctx) noexcept {
  ctx.error_out = nullptr;
}

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_configure {
  void operator()(const event::configure & ev, context & ctx) const noexcept {
    ctx.queue_ctx = ev.queue_ctx;
    ctx.try_dequeue = ev.try_dequeue;
    ctx.exporter_ctx = ev.exporter_ctx;
    ctx.flush_records = ev.flush_records;
    ctx.batch_capacity = ev.batch_capacity;
    ctx.tick_max_records = 0;
    ctx.batch_count = 0;
    ctx.error_out = ev.error_out;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct run_validate_config {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.queue_ctx == nullptr ||
        ctx.try_dequeue == nullptr ||
        ctx.flush_records == nullptr ||
        ctx.batch_capacity <= 0 ||
        ctx.batch_capacity > k_max_batch_capacity) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct begin_start {
  void operator()(const event::start & ev, context & ctx) const noexcept {
    ctx.batch_count = 0;
    ctx.tick_max_records = 0;
    ctx.error_out = ev.error_out;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct run_start {
  void operator()(context & ctx) const noexcept { run_validate_config{}(ctx); }
};

struct begin_tick {
  void operator()(const event::tick & ev, context & ctx) const noexcept {
    ctx.tick_max_records = ev.max_records;
    ctx.error_out = ev.error_out;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct run_collect_batch {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.queue_ctx == nullptr || ctx.try_dequeue == nullptr || ctx.batch_capacity <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    ctx.batch_count = 0;
    const int32_t limit =
      (ctx.tick_max_records > 0 && ctx.tick_max_records < ctx.batch_capacity)
        ? ctx.tick_max_records
        : ctx.batch_capacity;
    for (int32_t i = 0; i < limit; ++i) {
      emel::telemetry::record r = {};
      if (!ctx.try_dequeue(ctx.queue_ctx, &r)) {
        break;
      }
      ctx.batch[ctx.batch_count] = r;
      ctx.batch_count += 1;
    }
  }
};

struct run_flush_batch {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.batch_count <= 0) {
      return;
    }
    if (ctx.flush_records == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t flush_error = EMEL_OK;
    const bool ok = ctx.flush_records(
      ctx.exporter_ctx, ctx.batch.data(), ctx.batch_count, &flush_error);
    if (!ok || flush_error != EMEL_OK) {
      set_error(ctx, flush_error == EMEL_OK ? EMEL_ERR_BACKEND : flush_error);
      ctx.dropped_records += static_cast<uint64_t>(ctx.batch_count);
      ctx.batch_count = 0;
      return;
    }
    ctx.flushed_records += static_cast<uint64_t>(ctx.batch_count);
    ctx.batch_count = 0;
  }
};

struct run_backoff {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.backoff_count += 1;
  }
};

struct begin_stop {
  void operator()(const event::stop & ev, context & ctx) const noexcept {
    ctx.tick_max_records = 0;
    ctx.error_out = ev.error_out;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct run_stop {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.batch_count = 0;
  }
};

struct run_reset {
  void operator()(const event::reset & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.batch_count = 0;
    ctx.tick_max_records = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    clear_error_out(ctx);
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    if (ctx.error_out != nullptr) {
      *ctx.error_out = EMEL_OK;
    }
    clear_error_out(ctx);
  }
};

struct publish_error {
  void operator()(context & ctx) const noexcept {
    int32_t err = ctx.last_error;
    if (err == EMEL_OK) {
      err = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
    }
    ctx.last_error = err;
    if (ctx.error_out != nullptr) {
      *ctx.error_out = err;
    }
    clear_error_out(ctx);
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr begin_configure begin_configure{};
inline constexpr run_validate_config run_validate_config{};
inline constexpr begin_start begin_start{};
inline constexpr run_start run_start{};
inline constexpr begin_tick begin_tick{};
inline constexpr run_collect_batch run_collect_batch{};
inline constexpr run_flush_batch run_flush_batch{};
inline constexpr run_backoff run_backoff{};
inline constexpr begin_stop begin_stop{};
inline constexpr run_stop run_stop{};
inline constexpr run_reset run_reset{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::telemetry::exporter::action
