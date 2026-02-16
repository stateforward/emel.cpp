#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/batch/splitter/sm.hpp"
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

  emel::batch::splitter::sm batch_splitter = {};
  emel::memory::coordinator::sm memory_coordinator = {};
  emel::kv::cache::sm kv_cache = {};
  emel::decoder::ubatch_executor::sm ubatch_executor = {};
};

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

inline constexpr auto begin_decode = [](const event::decode & ev, context & ctx) {
  ctx.token_ids = ev.token_ids;
  ctx.n_tokens = ev.n_tokens;
  ctx.n_ubatch = ev.n_ubatch;

  ctx.outputs_total = 0;
  ctx.outputs_processed = 0;
  ctx.ubatches_total = 0;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes.fill(0);
  ctx.slot_offsets.fill(0);
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;  // GCOVR_EXCL_LINE
  *ev.error_out = EMEL_OK;

  if (ctx.n_tokens <= 0 || ctx.token_ids == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
};

inline constexpr auto run_initialize_batch = [](const event::initialize_batch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  int32_t outputs_total = 0;
  int32_t ubatch_count = 0;

  const bool ok = ctx.batch_splitter.process_event(emel::batch::splitter::event::split{
    .token_ids = ctx.token_ids,
    .n_tokens = ctx.n_tokens,
    .n_ubatch = ctx.n_ubatch,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ctx.ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ctx.ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &outputs_total,
  });

  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }

  if (ubatch_count <= 0 || outputs_total <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }

  ctx.ubatches_total = ubatch_count;
  ctx.ubatches_processed = 0;
  ctx.outputs_total = outputs_total;
  ctx.outputs_processed = 0;
  if (ctx.n_ubatch <= 0) {
    ctx.n_ubatch = ctx.n_tokens;
  }
};

inline constexpr auto run_update_memory = [](const event::update_memory & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::success;
  const bool ok = ctx.memory_coordinator.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
  });

  if (!ok || update_status_is_error(status)) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }
};

inline constexpr auto run_prepare_memory_batch = [](const event::prepare_memory_batch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ev.retryable_out != nullptr) {
    *ev.retryable_out = false;
  }

  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::success;
  const bool memory_ok = ctx.memory_coordinator.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = ctx.n_ubatch,
    .n_ubatches_total = ctx.ubatches_total,
    .status_out = &status,
  });

  if (!memory_ok) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }

  const prepare_failure_kind prepare_failure = classify_prepare_failure_from_memory_status(status);
  if (prepare_failure == prepare_failure_kind::retryable) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    if (ev.retryable_out != nullptr) {  // GCOVR_EXCL_LINE
      *ev.retryable_out = true;  // GCOVR_EXCL_LINE
    }
    return;  // GCOVR_EXCL_LINE
  }
  if (prepare_failure == prepare_failure_kind::permanent) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }

  const bool kv_ok = ctx.kv_cache.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ctx.ubatch_sizes.data(),
    .ubatch_count = ctx.ubatches_total,
    .requested_capacity = ctx.n_tokens,
    .slot_offsets_out = ctx.slot_offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(ctx.slot_offsets.size()),
    .ubatch_count_out = nullptr,
  });

  if (!kv_ok) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }
};

inline constexpr auto run_optimize_memory = [](const event::optimize_memory & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::success;
  const bool ok = ctx.memory_coordinator.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = true,
    .status_out = &status,
  });

  if (!ok) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }
};

inline constexpr auto run_reserve_output = [](const event::reserve_output & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ctx.outputs_total < 0) {
    *ev.error_out = EMEL_ERR_BACKEND;
  }
};

inline constexpr auto run_process_ubatch = [](const event::process_ubatch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.ubatches_processed >= ctx.ubatches_total) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  const int32_t current = ctx.ubatch_sizes[ctx.ubatches_processed];
  int32_t produced = 0;
  int32_t kv_tokens = 0;
  bool rollback_attempted = false;
  int32_t ubatch_error = EMEL_OK;
  const bool ok = ctx.ubatch_executor.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = ctx.ubatches_processed,
    .ubatch_size = current,
    .memory_coordinator_sm = &ctx.memory_coordinator,
    .kv_cache_sm = &ctx.kv_cache,
    .outputs_produced_out = &produced,
    .kv_tokens_out = &kv_tokens,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &ubatch_error,
  });
  if (!ok || ubatch_error != EMEL_OK) {
    const int32_t normalized = (ubatch_error == EMEL_OK || ubatch_error == EMEL_ERR_INVALID_ARGUMENT)
        ? EMEL_ERR_BACKEND
        : ubatch_error;
    *ev.error_out = normalized;  // GCOVR_EXCL_LINE
    if (ev.rollback_needed_out != nullptr) {  // GCOVR_EXCL_LINE
      *ev.rollback_needed_out = !rollback_attempted;  // GCOVR_EXCL_LINE
    }
    return;  // GCOVR_EXCL_LINE
  }

  if (produced <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    return;  // GCOVR_EXCL_LINE
  }

  if (ev.rollback_needed_out != nullptr) {
    *ev.rollback_needed_out = false;
  }
  ctx.outputs_processed += produced;
  ctx.ubatches_processed += 1;
};

inline constexpr auto run_rollback_ubatch = [](const event::rollback_ubatch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (!ev.rollback_needed) {
    return;  // GCOVR_EXCL_LINE
  }

  const int32_t rollback_to = std::max<int32_t>(0, ctx.ubatches_processed - 1);
  const bool kv_ok = ctx.kv_cache.process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = rollback_to,
  });
  if (!kv_ok) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;  // GCOVR_EXCL_LINE
  }

  if (ctx.outputs_processed > ctx.outputs_total) {
    *ev.error_out = EMEL_ERR_BACKEND;
  }
};

inline constexpr auto run_finalize_outputs = [](const event::finalize_outputs & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
  if (ctx.outputs_processed != ctx.outputs_total) {
    *ev.error_out = EMEL_ERR_BACKEND;
  }
};

inline constexpr auto dispatch_decoding_done_to_owner = [](const events::decoding_done & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.dispatch_event != nullptr) {
    (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                             .type = events::owner_event::kind::done,
                                             .done = ev,
                                         });
  }
};

inline constexpr auto dispatch_decoding_error_to_owner = [](const events::decoding_error & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out != nullptr) {
    *ev.error_out = ev.err == EMEL_OK ? EMEL_ERR_BACKEND : ev.err;
  }
  if (ev.dispatch_event != nullptr) {
    (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                             .type = events::owner_event::kind::error,
                                             .error = ev,
                                         });
  }
};

}  // namespace emel::decoder::action
