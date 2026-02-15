#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/events.hpp"
#include "emel/emel.h"

namespace emel::decoder::compute_executor::action {

struct context {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t kv_tokens = 0;
  int32_t * outputs_produced_out = nullptr;

  int32_t outputs_produced = 0;
  int32_t status_code = EMEL_OK;
};

inline constexpr auto begin_execute = [](const event::execute & ev, context & ctx) {
  ctx.ubatch_index = ev.ubatch_index;
  ctx.ubatch_size = ev.ubatch_size;
  ctx.kv_tokens = ev.kv_tokens;
  ctx.outputs_produced_out = ev.outputs_produced_out;
  ctx.outputs_produced = 0;
  ctx.status_code = EMEL_OK;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.ubatch_index < 0 || ctx.ubatch_size <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
    return;
  }
  if (ctx.kv_tokens < 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto run_bind_inputs = [](const event::bind_inputs & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_backend = [](const event::run_backend & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.kv_tokens <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto run_extract_outputs = [](const event::extract_outputs & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.kv_tokens < ctx.ubatch_size) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
    return;
  }

  ctx.outputs_produced = ctx.ubatch_size;
  if (ctx.outputs_produced_out != nullptr) {
    *ctx.outputs_produced_out = ctx.outputs_produced;
  }
};

inline constexpr auto on_compute_done = [](const events::compute_done &, context & ctx) {
  ctx.status_code = EMEL_OK;
};

inline constexpr auto on_compute_error = [](const events::compute_error & ev, context & ctx) {
  ctx.status_code = ev.err;
};

}  // namespace emel::decoder::compute_executor::action
