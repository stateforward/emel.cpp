#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/events.hpp"
#include "emel/emel.h"

namespace emel::decoder::compute_executor::action {

struct context {
  int32_t outputs_produced = 0;
  int32_t ubatch_index = 0;
  int32_t ubatch_size = 0;
  int32_t kv_tokens = 0;
  void * compute_ctx = nullptr;
  event::validate_fn validate = nullptr;
  event::prepare_graph_fn prepare_graph = nullptr;
  event::alloc_graph_fn alloc_graph = nullptr;
  event::bind_inputs_fn bind_inputs = nullptr;
  event::run_backend_fn run_backend = nullptr;
  event::extract_outputs_fn extract_outputs = nullptr;
  bool graph_reused = false;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

inline constexpr auto begin_execute = [](const event::execute & ev, context & ctx) {
  ctx.ubatch_index = ev.ubatch_index;
  ctx.ubatch_size = ev.ubatch_size;
  ctx.kv_tokens = ev.kv_tokens;
  ctx.compute_ctx = ev.compute_ctx;
  ctx.validate = ev.validate;
  ctx.prepare_graph = ev.prepare_graph;
  ctx.alloc_graph = ev.alloc_graph;
  ctx.bind_inputs = ev.bind_inputs;
  ctx.run_backend = ev.run_backend;
  ctx.extract_outputs = ev.extract_outputs;
  ctx.outputs_produced = 0;
  ctx.graph_reused = false;
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  if (request->validate == nullptr) {
    return;
  }

  int32_t err = EMEL_OK;
  const bool ok = request->validate(*request, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
  }
};

inline constexpr auto run_prepare_graph = [](const event::prepare_graph & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  bool reused = false;
  int32_t err = EMEL_OK;
  const bool ok = request->prepare_graph(*request, &reused, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
    return;
  }
  *ev.reused_out = reused;
};

inline constexpr auto run_alloc_graph = [](const event::alloc_graph & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  int32_t err = EMEL_OK;
  const bool ok = request->alloc_graph(*request, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
  }
};

inline constexpr auto run_bind_inputs = [](const event::bind_inputs & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  int32_t err = EMEL_OK;
  const bool ok = request->bind_inputs(*request, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
  }
};

inline constexpr auto run_backend = [](const event::run_backend & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  int32_t err = EMEL_OK;
  const bool ok = request->run_backend(*request, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
  }
};

inline constexpr auto run_extract_outputs = [](const event::extract_outputs & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  int32_t outputs_produced = 0;
  int32_t err = EMEL_OK;
  const bool ok = request->extract_outputs(*request, &outputs_produced, &err);
  if (!ok || err != EMEL_OK) {
    *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
    return;
  }
  ctx.outputs_produced = outputs_produced;
};

inline constexpr auto on_compute_done = [](const events::compute_done & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto on_compute_error = [](const events::compute_error & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = ev.err == EMEL_OK ? EMEL_ERR_BACKEND : ev.err;
  }
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

inline constexpr auto reject_invalid_validate = [](const event::validate & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_prepare_graph =
  [](const event::prepare_graph & ev, context &) {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  };

inline constexpr auto reject_invalid_alloc_graph = [](const event::alloc_graph & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_bind_inputs = [](const event::bind_inputs & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_run_backend = [](const event::run_backend & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto reject_invalid_extract_outputs =
  [](const event::extract_outputs & ev, context &) {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  };

inline constexpr auto reject_invalid_execute = [](context & ctx) noexcept {
  ctx.outputs_produced = 0;
  ctx.graph_reused = false;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto run_validate_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::validate validate{
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_validate(validate, ctx);
};

inline constexpr auto run_prepare_graph_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  ctx.graph_reused = false;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::prepare_graph prepare{
    .request = &request,
    .reused_out = &ctx.graph_reused,
    .error_out = &ctx.phase_error,
  };
  run_prepare_graph(prepare, ctx);
};

inline constexpr auto run_alloc_graph_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::alloc_graph alloc{
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_alloc_graph(alloc, ctx);
};

inline constexpr auto run_bind_inputs_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::bind_inputs bind{
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_bind_inputs(bind, ctx);
};

inline constexpr auto run_backend_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::run_backend run{
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_backend(run, ctx);
};

inline constexpr auto run_extract_outputs_phase = [](context & ctx) noexcept {
  ctx.phase_error = EMEL_OK;
  event::execute request{
    .ubatch_index = ctx.ubatch_index,
    .ubatch_size = ctx.ubatch_size,
    .kv_tokens = ctx.kv_tokens,
    .compute_ctx = ctx.compute_ctx,
    .validate = ctx.validate,
    .prepare_graph = ctx.prepare_graph,
    .alloc_graph = ctx.alloc_graph,
    .bind_inputs = ctx.bind_inputs,
    .run_backend = ctx.run_backend,
    .extract_outputs = ctx.extract_outputs,
  };
  event::extract_outputs extract{
    .request = &request,
    .error_out = &ctx.phase_error,
  };
  run_extract_outputs(extract, ctx);
};

inline constexpr auto mark_done = [](context & ctx) noexcept {
  ctx.last_error = EMEL_OK;
};

inline constexpr auto ensure_last_error = [](context & ctx) noexcept {
  if (ctx.last_error != EMEL_OK) {
    return;
  }
  ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
};

}  // namespace emel::decoder::compute_executor::action
