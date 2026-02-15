#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/sm.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace emel::decoder::ubatch_executor::action {

struct context {
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  emel::memory::coordinator::sm * memory_coordinator_sm = nullptr;
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  int32_t * outputs_produced_out = nullptr;
  int32_t * kv_tokens_out = nullptr;

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  int32_t status_code = EMEL_OK;
  bool rollback_attempted = false;
  emel::decoder::compute_executor::sm compute_executor = {};
};

inline bool prepare_status_is_error(const emel::memory::coordinator::event::memory_status status) {
  switch (status) {
    case emel::memory::coordinator::event::memory_status::success:
    case emel::memory::coordinator::event::memory_status::no_update:
      return false;
    case emel::memory::coordinator::event::memory_status::failed_prepare:
    case emel::memory::coordinator::event::memory_status::failed_compute:
      return true;
  }
  return true;
}

inline constexpr auto begin_execute = [](const event::execute & ev, context & ctx) {
  ctx.ubatch_index = ev.ubatch_index;
  ctx.ubatch_size = ev.ubatch_size;
  ctx.memory_coordinator_sm = ev.memory_coordinator_sm;
  ctx.kv_cache_sm = ev.kv_cache_sm;
  ctx.outputs_produced_out = ev.outputs_produced_out;
  ctx.kv_tokens_out = ev.kv_tokens_out;
  ctx.outputs_produced = 0;
  ctx.kv_tokens = 0;
  ctx.status_code = EMEL_OK;
  ctx.rollback_attempted = false;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.ubatch_index < 0 || ctx.ubatch_size <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
    return;
  }

  if (ctx.memory_coordinator_sm == nullptr || ctx.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto run_prepare_memory = [](const event::prepare_memory & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.memory_coordinator_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
    return;
  }

  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::success;
  const bool ok = ctx.memory_coordinator_sm->process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
  });

  if (!ok || prepare_status_is_error(status)) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto run_prepare_kv = [](const event::prepare_kv & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto run_compute = [](const event::run_compute & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
    return;
  }

  const bool ok = ctx.kv_cache_sm->process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = ctx.ubatch_index,
    .kv_tokens_out = &ctx.kv_tokens,
  });
  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
    return;
  }

  int32_t outputs_produced = 0;
  const bool compute_ok = ctx.compute_executor.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .outputs_produced_out = &outputs_produced,
  });
  if (!compute_ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = ctx.compute_executor.status_code();
    if (ctx.status_code == EMEL_OK) {
      ctx.status_code = EMEL_ERR_BACKEND;
    }
    return;
  }
  ctx.outputs_produced = outputs_produced;

  if (ctx.kv_tokens_out != nullptr) {
    *ctx.kv_tokens_out = ctx.kv_tokens;
  }
};

inline constexpr auto run_extract_outputs = [](const event::extract_outputs & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.outputs_produced <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
    return;
  }

  if (ctx.outputs_produced_out != nullptr) {
    *ctx.outputs_produced_out = ctx.outputs_produced;
  }
};

inline constexpr auto run_rollback = [](const event::rollback & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  ctx.rollback_attempted = true;

  if (ctx.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.status_code = *ev.error_out;
    return;
  }

  const bool ok = ctx.kv_cache_sm->process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = ctx.ubatch_index,
  });
  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.status_code = *ev.error_out;
  }
};

inline constexpr auto on_ubatch_execution_done = [](const events::ubatch_execution_done &, context & ctx) {
  ctx.status_code = EMEL_OK;
};

inline constexpr auto on_ubatch_execution_error = [](const events::ubatch_execution_error & ev, context & ctx) {
  ctx.status_code = ev.err;
};

}  // namespace emel::decoder::ubatch_executor::action
