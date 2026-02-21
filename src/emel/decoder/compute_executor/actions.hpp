#pragma once

#include "emel/decoder/compute_executor/context.hpp"

namespace emel::decoder::compute_executor::action {

namespace detail {

inline int32_t normalize_callback_error(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err != EMEL_OK) {
    return err;
  }
  return EMEL_ERR_BACKEND;
}

inline event::execute make_request(const context & ctx) noexcept {
  return event::execute{
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
    .outputs_produced_out = nullptr,
    .error_out = nullptr,
  };
}

}  // namespace detail

struct begin_execute {
  void operator()(const event::execute & ev, context & ctx) const noexcept {
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
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    if (request->validate == nullptr) {
      ctx.phase_error = EMEL_OK;
      return;
    }

    int32_t err = EMEL_OK;
    const bool ok = request->validate(*request, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    *ev.error_out = normalized;
    ctx.phase_error = normalized;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.validate == nullptr) {
      return;
    }
    event::execute request = detail::make_request(ctx);
    int32_t err = EMEL_OK;
    const bool ok = ctx.validate(request, &err);
    ctx.phase_error = detail::normalize_callback_error(ok, err);
  }
};

struct run_prepare_graph {
  void operator()(const event::prepare_graph & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr || request->prepare_graph == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    bool reused = false;
    int32_t err = EMEL_OK;
    const bool ok = request->prepare_graph(*request, &reused, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    if (ev.reused_out != nullptr) {
      *ev.reused_out = reused;
    }
    *ev.error_out = normalized;
    ctx.graph_reused = reused;
    ctx.phase_error = normalized;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.graph_reused = false;
    ctx.phase_error = EMEL_OK;
    if (ctx.prepare_graph == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    event::execute request = detail::make_request(ctx);
    bool reused = false;
    int32_t err = EMEL_OK;
    const bool ok = ctx.prepare_graph(request, &reused, &err);
    ctx.graph_reused = reused;
    ctx.phase_error = detail::normalize_callback_error(ok, err);
  }
};

struct run_alloc_graph {
  void operator()(const event::alloc_graph & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr || request->alloc_graph == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    int32_t err = EMEL_OK;
    const bool ok = request->alloc_graph(*request, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    *ev.error_out = normalized;
    ctx.phase_error = normalized;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.alloc_graph == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    event::execute request = detail::make_request(ctx);
    int32_t err = EMEL_OK;
    const bool ok = ctx.alloc_graph(request, &err);
    ctx.phase_error = detail::normalize_callback_error(ok, err);
  }
};

struct run_bind_inputs {
  void operator()(const event::bind_inputs & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr || request->bind_inputs == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    int32_t err = EMEL_OK;
    const bool ok = request->bind_inputs(*request, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    *ev.error_out = normalized;
    ctx.phase_error = normalized;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.bind_inputs == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    event::execute request = detail::make_request(ctx);
    int32_t err = EMEL_OK;
    const bool ok = ctx.bind_inputs(request, &err);
    ctx.phase_error = detail::normalize_callback_error(ok, err);
  }
};

struct run_backend {
  void operator()(const event::run_backend & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr || request->run_backend == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    int32_t err = EMEL_OK;
    const bool ok = request->run_backend(*request, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    *ev.error_out = normalized;
    ctx.phase_error = normalized;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.run_backend == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    event::execute request = detail::make_request(ctx);
    int32_t err = EMEL_OK;
    const bool ok = ctx.run_backend(request, &err);
    ctx.phase_error = detail::normalize_callback_error(ok, err);
  }
};

struct run_extract_outputs {
  void operator()(const event::extract_outputs & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;

    const event::execute * request = ev.request;
    if (request == nullptr || request->extract_outputs == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    int32_t outputs_produced = 0;
    int32_t err = EMEL_OK;
    const bool ok = request->extract_outputs(*request, &outputs_produced, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    if (normalized != EMEL_OK) {
      *ev.error_out = normalized;
      ctx.phase_error = normalized;
      return;
    }
    ctx.outputs_produced = outputs_produced;
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.extract_outputs == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    event::execute request = detail::make_request(ctx);
    int32_t outputs_produced = 0;
    int32_t err = EMEL_OK;
    const bool ok = ctx.extract_outputs(request, &outputs_produced, &err);
    const int32_t normalized = detail::normalize_callback_error(ok, err);
    if (normalized != EMEL_OK) {
      ctx.phase_error = normalized;
      return;
    }
    ctx.outputs_produced = outputs_produced;
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
  }
};

struct reject_invalid_execute {
  template <class event>
  void operator()(const event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
    ctx.outputs_produced = 0;
    ctx.graph_reused = false;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

inline constexpr begin_execute begin_execute{};
inline constexpr run_validate run_validate{};
inline constexpr run_prepare_graph run_prepare_graph{};
inline constexpr run_alloc_graph run_alloc_graph{};
inline constexpr run_bind_inputs run_bind_inputs{};
inline constexpr run_backend run_backend{};
inline constexpr run_extract_outputs run_extract_outputs{};
inline constexpr on_unexpected on_unexpected{};
inline constexpr reject_invalid_execute reject_invalid_execute{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};

}  // namespace emel::decoder::compute_executor::action
