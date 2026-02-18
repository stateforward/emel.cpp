#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

#include "emel/batch/splitter/sm.hpp"
#include "emel/callback.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace emel::decoder::action {

enum class prepare_failure_kind : uint8_t {
  none = 0,
  retryable,
  permanent,
};

struct context {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t n_ubatch = 0;

  int32_t outputs_total = 0;
  int32_t outputs_processed = 0;

  std::array<int32_t, emel::batch::splitter::action::MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, emel::kv::cache::action::MAX_UBATCHES> slot_offsets = {};
  int32_t ubatches_total = 0;
  int32_t ubatches_processed = 0;

  std::unique_ptr<emel::batch::splitter::sm> batch_splitter;
  std::unique_ptr<emel::memory::coordinator::sm> memory_coordinator;
  std::unique_ptr<emel::kv::cache::sm> kv_cache;
  std::unique_ptr<emel::decoder::ubatch_executor::sm> ubatch_executor;

  int32_t phase_error = EMEL_OK;
  int32_t ubatch_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  bool phase_retryable = false;
  bool rollback_needed = false;

  context();
};

inline context::context()
    : batch_splitter(std::make_unique<emel::batch::splitter::sm>()),
      memory_coordinator(std::make_unique<emel::memory::coordinator::sm>()),
      kv_cache(std::make_unique<emel::kv::cache::sm>()),
      ubatch_executor(std::make_unique<emel::decoder::ubatch_executor::sm>()) {
  // One-time heap allocation keeps decoder context small on the stack.
}

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

inline int32_t normalize_ubatch_error(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err == EMEL_OK || err == EMEL_ERR_INVALID_ARGUMENT) {
    return EMEL_ERR_BACKEND;
  }
  return err;
}

}  // namespace detail

inline prepare_failure_kind classify_prepare_failure_from_memory_status(
    const emel::memory::coordinator::event::memory_status status) {
  switch (status) {
    case emel::memory::coordinator::event::memory_status::success:
      return prepare_failure_kind::none;
    case emel::memory::coordinator::event::memory_status::failed_prepare:
      return prepare_failure_kind::retryable;
    case emel::memory::coordinator::event::memory_status::no_update:
    case emel::memory::coordinator::event::memory_status::failed_compute:
      return prepare_failure_kind::permanent;
  }
  return prepare_failure_kind::permanent;
}

inline bool update_status_is_error(
    const emel::memory::coordinator::event::memory_status status) {
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

struct begin_decode {
  void operator()(const event::decode & ev, context & ctx) const noexcept {
    ctx.token_ids = ev.token_ids;
    ctx.n_tokens = ev.n_tokens;
    ctx.n_ubatch = ev.n_ubatch;

    ctx.outputs_total = 0;
    ctx.outputs_processed = 0;
    ctx.ubatches_total = 0;
    ctx.ubatches_processed = 0;
    ctx.ubatch_sizes.fill(0);
    ctx.slot_offsets.fill(0);
    ctx.phase_error = EMEL_OK;
    ctx.ubatch_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.phase_retryable = false;
    ctx.rollback_needed = false;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;  // GCOVR_EXCL_LINE
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct reject_invalid_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct run_initialize_batch {
  void operator()(const event::initialize_batch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    struct split_reply {
      int32_t * ubatch_sizes_out = nullptr;
      int32_t ubatch_sizes_capacity = 0;
      int32_t ubatch_count = 0;
      int32_t total_outputs = 0;
      int32_t err = EMEL_OK;

      void on_done(const emel::batch::splitter::events::splitting_done & reply) noexcept {
        err = EMEL_OK;
        ubatch_count = reply.ubatch_count;
        total_outputs = reply.total_outputs;
        if (ubatch_sizes_out == nullptr) {
          return;  // GCOVR_EXCL_LINE
        }
        if (ubatch_sizes_capacity < reply.ubatch_count || reply.ubatch_sizes == nullptr) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        for (int32_t i = 0; i < reply.ubatch_count; ++i) {
          ubatch_sizes_out[i] = reply.ubatch_sizes[i];
        }
      }

      void on_error(const emel::batch::splitter::events::splitting_error & reply) noexcept {
        err = reply.err;
      }
    };

    split_reply reply{
      .ubatch_sizes_out = ctx.ubatch_sizes.data(),
      .ubatch_sizes_capacity = static_cast<int32_t>(ctx.ubatch_sizes.size()),
    };

    const auto on_done =
        emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
            split_reply, &split_reply::on_done>(&reply);
    const auto on_error =
        emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
            split_reply, &split_reply::on_error>(&reply);

    const bool ok = ctx.batch_splitter->process_event(emel::batch::splitter::event::split{
      .token_ids = ctx.token_ids,
      .n_tokens = ctx.n_tokens,
      .n_ubatch = ctx.n_ubatch,
      .mode = emel::batch::splitter::event::split_mode::simple,
      .on_done = on_done,
      .on_error = on_error,
    });

    if (!ok || reply.err != EMEL_OK) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    if (reply.ubatch_count <= 0 || reply.total_outputs <= 0) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    ctx.ubatches_total = reply.ubatch_count;
    ctx.ubatches_processed = 0;
    ctx.outputs_total = reply.total_outputs;
    ctx.outputs_processed = 0;
    if (ctx.n_ubatch <= 0) {
      ctx.n_ubatch = ctx.n_tokens;
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::initialize_batch initialize{
      .error_out = &ctx.phase_error,
    };
    (*this)(initialize, ctx);
  }
};

struct run_update_memory {
  void operator()(const event::update_memory & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (ctx.memory_coordinator == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    emel::memory::coordinator::event::memory_status status =
        emel::memory::coordinator::event::memory_status::success;
    const bool ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::prepare_update{
      .optimize = false,
      .status_out = &status,
    });

    if (!ok || update_status_is_error(status)) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::update_memory update{
      .error_out = &ctx.phase_error,
    };
    (*this)(update, ctx);
  }
};

struct run_prepare_memory_batch {
  void operator()(const event::prepare_memory_batch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
    ctx.phase_retryable = false;
    if (ev.retryable_out != nullptr) {
      *ev.retryable_out = false;
    }

    if (ctx.memory_coordinator == nullptr || ctx.kv_cache == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    emel::memory::coordinator::event::memory_status status =
        emel::memory::coordinator::event::memory_status::success;
    const bool memory_ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::prepare_batch{
      .n_ubatch = ctx.n_ubatch,
      .n_ubatches_total = ctx.ubatches_total,
      .status_out = &status,
    });

    if (!memory_ok) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    const prepare_failure_kind prepare_failure = classify_prepare_failure_from_memory_status(status);
    if (prepare_failure == prepare_failure_kind::retryable) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_retryable = true;  // GCOVR_EXCL_LINE
      if (ev.retryable_out != nullptr) {  // GCOVR_EXCL_LINE
        *ev.retryable_out = true;  // GCOVR_EXCL_LINE
      }  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
    if (prepare_failure == prepare_failure_kind::permanent) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    const bool kv_ok = ctx.kv_cache->process_event(emel::kv::cache::event::prepare{
      .ubatch_sizes = ctx.ubatch_sizes.data(),
      .ubatch_count = ctx.ubatches_total,
      .requested_capacity = ctx.n_tokens,
      .slot_offsets_out = ctx.slot_offsets.data(),
      .slot_offsets_capacity = static_cast<int32_t>(ctx.slot_offsets.size()),
      .ubatch_count_out = nullptr,
    });

    if (!kv_ok) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.phase_retryable = false;
    event::prepare_memory_batch prepare{
      .error_out = &ctx.phase_error,
      .retryable_out = &ctx.phase_retryable,
    };
    (*this)(prepare, ctx);
  }
};

struct run_optimize_memory {
  void operator()(const event::optimize_memory & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (ctx.memory_coordinator == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    emel::memory::coordinator::event::memory_status status =
        emel::memory::coordinator::event::memory_status::success;
    const bool ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::prepare_update{
      .optimize = true,
      .status_out = &status,
    });

    if (!ok) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::optimize_memory optimize{
      .error_out = &ctx.phase_error,
    };
    (*this)(optimize, ctx);
  }
};

struct run_reserve_output {
  void operator()(const event::reserve_output & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct reject_invalid_reserve_output {
  void operator()(const event::reserve_output & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.phase_error = EMEL_ERR_BACKEND;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
  }
};

struct run_process_ubatch {
  void operator()(const event::process_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.rollback_needed_out != nullptr) {
      *ev.rollback_needed_out = false;
    }
    ctx.phase_error = EMEL_OK;
    ctx.rollback_needed = false;

    if (ctx.ubatch_executor == nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      ctx.ubatch_error = EMEL_ERR_BACKEND;
      ctx.rollback_needed = true;
      if (ev.rollback_needed_out != nullptr) {
        *ev.rollback_needed_out = true;
      }
      return;
    }

    if (ctx.ubatches_processed < 0 || ctx.ubatches_processed >= ctx.ubatches_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      ctx.ubatch_error = EMEL_ERR_BACKEND;
      return;
    }

    const int32_t current = ctx.ubatch_sizes[ctx.ubatches_processed];
    int32_t produced = 0;
    int32_t kv_tokens = 0;
    bool rollback_attempted = false;
    int32_t ubatch_error = EMEL_OK;
    const bool ok = ctx.ubatch_executor->process_event(emel::decoder::ubatch_executor::event::execute{
      .ubatch_index = ctx.ubatches_processed,
      .ubatch_size = current,
      .memory_coordinator_sm = ctx.memory_coordinator.get(),
      .kv_cache_sm = ctx.kv_cache.get(),
      .outputs_produced_out = &produced,
      .kv_tokens_out = &kv_tokens,
      .rollback_attempted_out = &rollback_attempted,
      .error_out = &ubatch_error,
    });

    const int32_t normalized = detail::normalize_ubatch_error(ok, ubatch_error);
    if (normalized != EMEL_OK) {
      *ev.error_out = normalized;  // GCOVR_EXCL_LINE
      ctx.phase_error = normalized;  // GCOVR_EXCL_LINE
      ctx.ubatch_error = normalized;  // GCOVR_EXCL_LINE
      const bool rollback_needed = !rollback_attempted;
      if (ev.rollback_needed_out != nullptr) {  // GCOVR_EXCL_LINE
        *ev.rollback_needed_out = rollback_needed;  // GCOVR_EXCL_LINE
      }
      ctx.rollback_needed = rollback_needed;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    if (produced <= 0) {  // GCOVR_EXCL_LINE
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.ubatch_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    ctx.outputs_processed += produced;
    ctx.ubatches_processed += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.rollback_needed = false;
    event::process_ubatch process{
      .error_out = &ctx.phase_error,
      .rollback_needed_out = &ctx.rollback_needed,
    };
    (*this)(process, ctx);
    if (ctx.phase_error != EMEL_OK) {
      ctx.ubatch_error = ctx.phase_error;
    }
  }
};

struct on_invalid_ubatch_size {
  void operator()(const event::process_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
    }
    if (ev.rollback_needed_out != nullptr) {
      *ev.rollback_needed_out = false;
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.ubatch_error = EMEL_ERR_BACKEND;
    ctx.rollback_needed = false;
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.ubatch_error = EMEL_ERR_BACKEND;
    ctx.rollback_needed = false;
  }
};

struct run_rollback_ubatch {
  void operator()(const event::rollback_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (!ev.rollback_needed) {
      return;  // GCOVR_EXCL_LINE
    }

    if (ctx.kv_cache == nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    const int32_t rollback_to = std::max<int32_t>(0, ctx.ubatches_processed - 1);
    int32_t kv_error = EMEL_OK;
    const bool kv_ok = ctx.kv_cache->process_event(emel::kv::cache::event::rollback{
      .from_ubatch_index = rollback_to,
      .error_out = &kv_error,
    });
    const int32_t normalized = detail::normalize_error(kv_ok, kv_error);
    if (normalized != EMEL_OK) {
      *ev.error_out = normalized;
      ctx.phase_error = normalized;
      return;  // GCOVR_EXCL_LINE
    }

    if (ctx.outputs_processed > ctx.outputs_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::rollback_ubatch rollback{
      .error_out = &ctx.phase_error,
      .rollback_needed = ctx.rollback_needed,
    };
    (*this)(rollback, ctx);
  }
};

struct run_finalize_outputs {
  void operator()(const event::finalize_outputs & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
    if (ctx.outputs_processed != ctx.outputs_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
    }
  }

  template <class Ev>
  void operator()(const Ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::finalize_outputs finalize{
      .error_out = &ctx.phase_error,
    };
    (*this)(finalize, ctx);
  }
};

struct dispatch_decoding_done_to_owner {
  void operator()(const events::decoding_done & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.dispatch_event != nullptr) {
      (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::done,
                                               .done = ev,
                                           });
    }
  }
};

struct dispatch_decoding_error_to_owner {
  void operator()(const events::decoding_error & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = ev.err == EMEL_OK ? EMEL_ERR_BACKEND : ev.err;
    }
    if (ev.dispatch_event != nullptr) {
      (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::error,
                                               .error = ev,
                                           });
    }
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

struct capture_ubatch_error {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = ctx.ubatch_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.ubatch_error;
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

inline constexpr begin_decode begin_decode{};
inline constexpr run_validate run_validate{};
inline constexpr reject_invalid_validate reject_invalid_validate{};
inline constexpr run_initialize_batch run_initialize_batch{};
inline constexpr run_update_memory run_update_memory{};
inline constexpr run_prepare_memory_batch run_prepare_memory_batch{};
inline constexpr run_optimize_memory run_optimize_memory{};
inline constexpr run_reserve_output run_reserve_output{};
inline constexpr reject_invalid_reserve_output reject_invalid_reserve_output{};
inline constexpr run_process_ubatch run_process_ubatch{};
inline constexpr on_invalid_ubatch_size on_invalid_ubatch_size{};
inline constexpr run_rollback_ubatch run_rollback_ubatch{};
inline constexpr run_finalize_outputs run_finalize_outputs{};
inline constexpr dispatch_decoding_done_to_owner dispatch_decoding_done_to_owner{};
inline constexpr dispatch_decoding_error_to_owner dispatch_decoding_error_to_owner{};
inline constexpr mark_done mark_done{};
inline constexpr capture_rollback_error capture_rollback_error{};
inline constexpr capture_ubatch_error capture_ubatch_error{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::decoder::action
