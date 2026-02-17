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

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
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
  ctx.outputs_produced = 0;
  ctx.kv_tokens = 0;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.ubatch_index < 0 || ctx.ubatch_size <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
};

inline constexpr auto run_prepare_memory = [](const event::prepare_memory & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ev.memory_coordinator_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::success;
  const bool ok = ev.memory_coordinator_sm->process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
  });

  if (!ok || prepare_status_is_error(status)) {
    *ev.error_out = EMEL_ERR_BACKEND;
  }
};

inline constexpr auto run_prepare_kv = [](const event::prepare_kv & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ev.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto run_compute = [](const event::run_compute & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ev.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  const event::execute * request = ev.request;
  if (request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  const bool ok = ev.kv_cache_sm->process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = ctx.ubatch_index,
    .kv_tokens_out = &ctx.kv_tokens,
  });
  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  int32_t outputs_produced = 0;
  int32_t compute_error = EMEL_OK;
  const bool compute_ok = ctx.compute_executor.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = request->compute_ctx,
    .validate = request->compute_validate,
    .prepare_graph = request->compute_prepare_graph,
    .alloc_graph = request->compute_alloc_graph,
    .bind_inputs = request->compute_bind_inputs,
    .run_backend = request->compute_run_backend,
    .extract_outputs = request->compute_extract_outputs,
    .outputs_produced_out = &outputs_produced,
    .error_out = &compute_error,
  });
  if (!compute_ok || compute_error != EMEL_OK) {
    *ev.error_out = compute_error == EMEL_OK ? EMEL_ERR_BACKEND : compute_error;
    return;
  }
  ctx.outputs_produced = outputs_produced;

};

inline constexpr auto run_extract_outputs = [](const event::extract_outputs & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.outputs_produced <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

};

inline constexpr auto run_rollback = [](const event::rollback & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.kv_cache_sm == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  const bool ok = ev.kv_cache_sm->process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = ctx.ubatch_index,
  });
  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
  }
};

inline constexpr auto on_ubatch_execution_done = [](const events::ubatch_execution_done & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto on_ubatch_execution_error = [](const events::ubatch_execution_error & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = ev.err == EMEL_OK ? EMEL_ERR_BACKEND : ev.err;
  }
};

}  // namespace emel::decoder::ubatch_executor::action
