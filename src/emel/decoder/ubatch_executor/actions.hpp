#pragma once

#include "emel/decoder/ubatch_executor/context.hpp"

namespace emel::decoder::ubatch_executor::action {

namespace detail {

inline int32_t normalize_error(const bool ok, const int32_t err) noexcept {
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
    .memory_coordinator_sm = ctx.memory_coordinator_sm,
    .kv_cache_sm = ctx.kv_cache_sm,
    .expected_outputs = ctx.expected_outputs,
    .compute_ctx = ctx.compute_ctx,
    .compute_validate = ctx.compute_validate,
    .compute_prepare_graph = ctx.compute_prepare_graph,
    .compute_alloc_graph = ctx.compute_alloc_graph,
    .compute_bind_inputs = ctx.compute_bind_inputs,
    .compute_run_backend = ctx.compute_run_backend,
    .compute_extract_outputs = ctx.compute_extract_outputs,
    .outputs_produced_out = nullptr,
    .kv_tokens_out = nullptr,
    .rollback_attempted_out = nullptr,
    .error_out = nullptr,
    .positions = ctx.positions,
    .positions_count = ctx.positions_count,
    .seq_masks = ctx.seq_masks,
    .seq_mask_words = ctx.seq_mask_words,
    .seq_masks_count = ctx.seq_masks_count,
    .seq_primary_ids = ctx.seq_primary_ids,
    .seq_primary_ids_count = ctx.seq_primary_ids_count,
  };
}

}  // namespace detail

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

struct begin_execute {
  void operator()(const event::execute & ev, context & ctx) const noexcept {
    ctx.ubatch_index = ev.ubatch_index;
    ctx.ubatch_size = ev.ubatch_size;
    ctx.expected_outputs = ev.expected_outputs;
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
    ctx.seq_masks = ev.seq_masks;
    ctx.seq_mask_words = ev.seq_mask_words;
    ctx.seq_masks_count = ev.seq_masks_count;
    ctx.seq_primary_ids = ev.seq_primary_ids;
    ctx.seq_primary_ids_count = ev.seq_primary_ids_count;
    ctx.phase_error = EMEL_OK;
    ctx.execution_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.rollback_attempted = false;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct run_prepare_memory {
  void operator()(const event::prepare_memory & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.memory_coordinator_sm == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.memory_coordinator_sm == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
  }
};

struct run_prepare_kv {
  void operator()(const event::prepare_kv & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.kv_cache_sm == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = ctx.kv_cache_sm == nullptr ? EMEL_ERR_INVALID_ARGUMENT : EMEL_OK;
  }
};

struct run_compute {
  void operator()(const event::run_compute & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.kv_cache_sm == nullptr || ev.request == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      ctx.execution_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    const event::execute * request = ev.request;
    int32_t kv_error = EMEL_OK;
    const bool ok = ev.kv_cache_sm->process_event(emel::kv::cache::event::apply_ubatch{
      .ubatch_index = ctx.ubatch_index,
      .kv_tokens_out = &ctx.kv_tokens,
      .error_out = &kv_error,
      .positions = request->positions,
      .positions_count = request->positions_count,
      .seq_masks = request->seq_masks,
      .seq_mask_words = request->seq_mask_words,
      .seq_masks_count = request->seq_masks_count,
      .seq_primary_ids = request->seq_primary_ids,
      .seq_primary_ids_count = request->seq_primary_ids_count,
    });
    if (!ok || kv_error != EMEL_OK) {
      const int32_t err = kv_error == EMEL_OK ? EMEL_ERR_BACKEND : kv_error;
      *ev.error_out = err;
      ctx.phase_error = err;
      ctx.execution_error = err;
      return;
    }

    int32_t outputs_produced = 0;
    int32_t compute_error = EMEL_OK;
    const bool compute_ok = ctx.compute_executor.process_event(
      emel::decoder::compute_executor::event::execute{
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
      const int32_t err = compute_error == EMEL_OK ? EMEL_ERR_BACKEND : compute_error;
      *ev.error_out = err;
      ctx.phase_error = err;
      ctx.execution_error = err;
      return;
    }

    ctx.outputs_produced = outputs_produced;
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.kv_cache_sm == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      ctx.execution_error = ctx.phase_error;
      return;
    }
    event::execute request = detail::make_request(ctx);
    event::run_compute run{
      .kv_cache_sm = ctx.kv_cache_sm,
      .request = &request,
      .error_out = &ctx.phase_error,
    };
    (*this)(run, ctx);
    if (ctx.phase_error != EMEL_OK) {
      ctx.execution_error = ctx.phase_error;
    }
  }
};

struct run_extract_outputs {
  void operator()(const event::extract_outputs & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct run_rollback {
  void operator()(const event::rollback & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.kv_cache_sm == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    int32_t kv_error = EMEL_OK;
    const bool ok = ev.kv_cache_sm->process_event(emel::kv::cache::event::rollback{
      .from_ubatch_index = ctx.ubatch_index,
      .error_out = &kv_error,
    });
    const int32_t err = detail::normalize_error(ok, kv_error);
    if (err != EMEL_OK) {
      *ev.error_out = err;
      ctx.phase_error = err;
      return;
    }
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.rollback_attempted = true;
    if (ctx.kv_cache_sm == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    int32_t kv_error = EMEL_OK;
    const bool ok = ctx.kv_cache_sm->process_event(emel::kv::cache::event::rollback{
      .from_ubatch_index = ctx.ubatch_index,
      .error_out = &kv_error,
    });
    const int32_t err = detail::normalize_error(ok, kv_error);
    if (err != EMEL_OK) {
      ctx.phase_error = err;
    }
  }
};

struct mark_missing_outputs {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.execution_error = EMEL_ERR_BACKEND;
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
    ctx.kv_tokens = 0;
    ctx.expected_outputs = 0;
    ctx.rollback_attempted = false;
    ctx.execution_error = EMEL_OK;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct capture_rollback_error {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct capture_execution_error {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = ctx.execution_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.execution_error;
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

inline constexpr begin_execute begin_execute{};
inline constexpr run_validate run_validate{};
inline constexpr run_prepare_memory run_prepare_memory{};
inline constexpr run_prepare_kv run_prepare_kv{};
inline constexpr run_compute run_compute{};
inline constexpr run_extract_outputs run_extract_outputs{};
inline constexpr run_rollback run_rollback{};
inline constexpr mark_missing_outputs mark_missing_outputs{};
inline constexpr reject_invalid_execute reject_invalid_execute{};
inline constexpr mark_done mark_done{};
inline constexpr capture_rollback_error capture_rollback_error{};
inline constexpr capture_execution_error capture_execution_error{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::decoder::ubatch_executor::action
