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
  emel::memory::coordinator::sm * memory_coordinator_sm = nullptr;
  emel::kv::cache::sm * kv_cache_sm = nullptr;
  void * compute_ctx = nullptr;
  event::compute_validate_fn compute_validate = nullptr;
  event::compute_prepare_graph_fn compute_prepare_graph = nullptr;
  event::compute_alloc_graph_fn compute_alloc_graph = nullptr;
  event::compute_bind_inputs_fn compute_bind_inputs = nullptr;
  event::compute_run_backend_fn compute_run_backend = nullptr;
  event::compute_extract_outputs_fn compute_extract_outputs = nullptr;
  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t execution_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  bool rollback_attempted = false;
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
  ctx.memory_coordinator_sm = ev.memory_coordinator_sm;
  ctx.kv_cache_sm = ev.kv_cache_sm;
  ctx.compute_ctx = ev.compute_ctx;
  ctx.compute_validate = ev.compute_validate;
  ctx.compute_prepare_graph = ev.compute_prepare_graph;
  ctx.compute_alloc_graph = ev.compute_alloc_graph;
  ctx.compute_bind_inputs = ev.compute_bind_inputs;
  ctx.compute_run_backend = ev.compute_run_backend;
  ctx.compute_extract_outputs = ev.compute_extract_outputs;
  ctx.positions = ev.positions;
  ctx.positions_count = ev.positions_count;
  ctx.phase_error = EMEL_OK;
  ctx.execution_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.rollback_attempted = false;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  (void)ctx;
};

inline constexpr auto run_prepare_memory = [](const event::prepare_memory & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

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
};

inline constexpr auto run_compute = [](const event::run_compute & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  const event::execute * request = ev.request;

  int32_t kv_error = EMEL_OK;
  const bool ok = ev.kv_cache_sm->process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = ctx.ubatch_index,
    .kv_tokens_out = &ctx.kv_tokens,
    .error_out = &kv_error,
    .positions = request->positions,
    .positions_count = request->positions_count,
  });
  if (!ok || kv_error != EMEL_OK) {
    *ev.error_out = kv_error == EMEL_OK ? EMEL_ERR_BACKEND : kv_error;
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
  (void)ctx;
};

inline constexpr auto run_rollback = [](const event::rollback & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
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

inline constexpr auto reject_invalid_validate = [](const event::validate & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_prepare_memory =
  [](const event::prepare_memory & ev, context &) {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  };

inline constexpr auto reject_invalid_prepare_kv = [](const event::prepare_kv & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_run_compute = [](const event::run_compute & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_extract_outputs =
  [](const event::extract_outputs & ev, context &) {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
    }
  };

inline constexpr auto reject_invalid_rollback = [](const event::rollback & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_execute = [](context & ctx) noexcept {
  ctx.outputs_produced = 0;
  ctx.kv_tokens = 0;
  ctx.rollback_attempted = false;
  ctx.execution_error = EMEL_OK;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto run_validate_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::validate validate{
    .request = nullptr,
    .error_out = &ctx.phase_error,
  };
  run_validate(validate, ctx);
};

inline constexpr auto run_prepare_memory_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::prepare_memory prepare{
    .memory_coordinator_sm = ctx.memory_coordinator_sm,
    .error_out = &ctx.phase_error,
  };
  run_prepare_memory(prepare, ctx);
};

inline constexpr auto run_prepare_kv_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::prepare_kv prepare{
    .kv_cache_sm = ctx.kv_cache_sm,
    .error_out = &ctx.phase_error,
  };
  run_prepare_kv(prepare, ctx);
};

inline constexpr auto run_compute_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .memory_coordinator_sm = ctx.memory_coordinator_sm,
    .kv_cache_sm = ctx.kv_cache_sm,
    .compute_ctx = ctx.compute_ctx,
    .compute_validate = ctx.compute_validate,
    .compute_prepare_graph = ctx.compute_prepare_graph,
    .compute_alloc_graph = ctx.compute_alloc_graph,
    .compute_bind_inputs = ctx.compute_bind_inputs,
    .compute_run_backend = ctx.compute_run_backend,
    .compute_extract_outputs = ctx.compute_extract_outputs,
    .positions = ctx.positions,
    .positions_count = ctx.positions_count,
  };
  event::run_compute run{
    .kv_cache_sm = ctx.kv_cache_sm,
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_compute(run, ctx);
  if (ctx.phase_error != EMEL_OK) {
    ctx.execution_error = ctx.phase_error;
  }
};

inline constexpr auto run_extract_outputs_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::extract_outputs extract{
    .error_out = &ctx.phase_error,
  };
  run_extract_outputs(extract, ctx);
  if (ctx.phase_error != EMEL_OK) {
    ctx.execution_error = ctx.phase_error;
  }
};

inline constexpr auto mark_missing_outputs = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.execution_error = EMEL_ERR_BACKEND;
};

inline constexpr auto run_rollback_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  ctx.rollback_attempted = true;
  event::rollback rollback{
    .kv_cache_sm = ctx.kv_cache_sm,
    .error_out = &ctx.phase_error,
  };
  run_rollback(rollback, ctx);
};

inline constexpr auto mark_done = [](context & ctx) noexcept {
  ctx.last_error = EMEL_OK;
};

inline constexpr auto capture_rollback_error = [](context & ctx) noexcept {
  ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
};

inline constexpr auto capture_execution_error = [](context & ctx) noexcept {
  ctx.last_error = ctx.execution_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.execution_error;
};

inline constexpr auto ensure_last_error = [](context & ctx) noexcept {
  if (ctx.last_error != EMEL_OK) {
    return;
  }
  ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
  }
};

}  // namespace emel::decoder::ubatch_executor::action
