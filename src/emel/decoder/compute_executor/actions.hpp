#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/events.hpp"
#include "emel/emel.h"

namespace emel::decoder::compute_executor::action {

struct context {
  int32_t outputs_produced = 0;
};

inline constexpr auto begin_execute = [](const event::execute & ev, context & ctx) {
  (void)ev;
  ctx.outputs_produced = 0;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::execute * request = ev.request;
  if (request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (request->prepare_graph == nullptr ||
      request->bind_inputs == nullptr ||
      request->run_backend == nullptr ||
      request->extract_outputs == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (request->ubatch_index < 0 || request->ubatch_size <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }
  if (request->kv_tokens < 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  if (request == nullptr || request->prepare_graph == nullptr || ev.reused_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  if (request == nullptr || request->alloc_graph == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  if (request == nullptr || request->bind_inputs == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  if (request == nullptr || request->run_backend == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  if (request == nullptr || request->extract_outputs == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
  void operator()(const Event & ev, context &) const {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

}  // namespace emel::decoder::compute_executor::action
