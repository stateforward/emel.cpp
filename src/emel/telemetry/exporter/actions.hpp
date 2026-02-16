#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/telemetry/exporter/events.hpp"

namespace emel::telemetry::exporter::action {

inline constexpr int32_t k_max_batch_capacity = 256;

struct context {
  void * queue_ctx = nullptr;
  emel::telemetry::dequeue_record_fn try_dequeue = nullptr;
  void * exporter_ctx = nullptr;
  emel::telemetry::flush_records_fn flush_records = nullptr;
  int32_t batch_capacity = 64;
  int32_t tick_max_records = 0;

  std::array<emel::telemetry::record, k_max_batch_capacity> batch = {};
  int32_t batch_count = 0;
  uint64_t flushed_records = 0;
  uint64_t dropped_records = 0;
  uint32_t backoff_count = 0;
};

inline constexpr auto begin_configure = [](const event::configure & ev, context & ctx) {
  ctx.queue_ctx = ev.queue_ctx;
  ctx.try_dequeue = ev.try_dequeue;
  ctx.exporter_ctx = ev.exporter_ctx;
  ctx.flush_records = ev.flush_records;
  ctx.batch_capacity = ev.batch_capacity;
  ctx.tick_max_records = 0;
  ctx.batch_count = 0;
};

inline constexpr auto run_validate_config = [](const event::validate_config & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (
      ctx.queue_ctx == nullptr ||
      ctx.try_dequeue == nullptr ||
      ctx.flush_records == nullptr ||
      ctx.batch_capacity <= 0 ||
      ctx.batch_capacity > k_max_batch_capacity) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto begin_start = [](const event::start &, context & ctx) {
  ctx.batch_count = 0;
  ctx.tick_max_records = 0;
};

inline constexpr auto run_start = [](const event::run_start & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (
      ctx.queue_ctx == nullptr ||
      ctx.try_dequeue == nullptr ||
      ctx.flush_records == nullptr ||
      ctx.batch_capacity <= 0 ||
      ctx.batch_capacity > k_max_batch_capacity) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto begin_tick = [](const event::tick & ev, context & ctx) {
  ctx.tick_max_records = ev.max_records;
};

inline constexpr auto run_collect_batch = [](const event::collect_batch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.queue_ctx == nullptr || ctx.try_dequeue == nullptr || ctx.batch_capacity <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  ctx.batch_count = 0;
  const int32_t limit = (ctx.tick_max_records > 0 && ctx.tick_max_records < ctx.batch_capacity)
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
};

inline constexpr auto run_flush_batch = [](const event::flush_batch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.batch_count <= 0) {
    return;
  }
  if (ctx.flush_records == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  int32_t flush_error = EMEL_OK;
  const bool ok = ctx.flush_records(
      ctx.exporter_ctx, ctx.batch.data(), ctx.batch_count, &flush_error);
  if (!ok || flush_error != EMEL_OK) {
    *ev.error_out = flush_error == EMEL_OK ? EMEL_ERR_BACKEND : flush_error;
    ctx.dropped_records += static_cast<uint64_t>(ctx.batch_count);
    ctx.batch_count = 0;
    return;
  }

  ctx.flushed_records += static_cast<uint64_t>(ctx.batch_count);
  ctx.batch_count = 0;
};

inline constexpr auto run_backoff = [](const event::run_backoff & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  ctx.backoff_count += 1;
};

inline constexpr auto begin_stop = [](const event::stop &, context & ctx) {
  ctx.tick_max_records = 0;
};

inline constexpr auto run_stop = [](const event::run_stop & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  ctx.batch_count = 0;
};

inline constexpr auto run_reset = [](const event::reset & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.batch_count = 0;
  ctx.tick_max_records = 0;
};

}  // namespace emel::telemetry::exporter::action
