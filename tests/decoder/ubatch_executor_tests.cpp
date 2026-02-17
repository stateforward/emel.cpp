#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/guards.hpp"
#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace {

using compute_execute_t = emel::decoder::compute_executor::event::execute;

bool compute_validate(const compute_execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.ubatch_index < 0 || ev.ubatch_size <= 0 || ev.kv_tokens < 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  return true;
}

bool compute_prepare_graph(const compute_execute_t &, bool * reused_out, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  return true;
}

bool compute_alloc_graph(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_bind_inputs(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_run_backend(const compute_execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = ev.kv_tokens > 0 ? EMEL_OK : EMEL_ERR_BACKEND;
  }
  return ev.kv_tokens > 0;
}

bool compute_extract_outputs(
    const compute_execute_t & ev, int32_t * outputs_out, int32_t * err_out) {
  if (ev.kv_tokens < ev.ubatch_size) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  }
  if (outputs_out != nullptr) {
    *outputs_out = ev.ubatch_size;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

void apply_compute_callbacks(emel::decoder::ubatch_executor::event::execute & ev) {
  ev.compute_validate = compute_validate;
  ev.compute_prepare_graph = compute_prepare_graph;
  ev.compute_alloc_graph = compute_alloc_graph;
  ev.compute_bind_inputs = compute_bind_inputs;
  ev.compute_run_backend = compute_run_backend;
  ev.compute_extract_outputs = compute_extract_outputs;
}

void apply_compute_callbacks(emel::decoder::compute_executor::event::execute & ev) {
  ev.validate = compute_validate;
  ev.prepare_graph = compute_prepare_graph;
  ev.alloc_graph = compute_alloc_graph;
  ev.bind_inputs = compute_bind_inputs;
  ev.run_backend = compute_run_backend;
  ev.extract_outputs = compute_extract_outputs;
}

bool prepare_kv(
    emel::kv::cache::sm & kv_cache,
    const int32_t * ubatch_sizes,
    const int32_t ubatch_count,
    const int32_t requested_capacity) {
  return kv_cache.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes,
    .ubatch_count = ubatch_count,
    .requested_capacity = requested_capacity,
    .slot_offsets_out = nullptr,
    .slot_offsets_capacity = 0,
    .ubatch_count_out = nullptr,
  });
}

}  // namespace

TEST_CASE("ubatch_executor_starts_initialized") {
  emel::decoder::ubatch_executor::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
}

TEST_CASE("ubatch_executor_execute_success_path") {
  emel::decoder::ubatch_executor::sm machine{};
    [[maybe_unused]] emel::memory::coordinator::sm memory_coordinator{};
    [[maybe_unused]] emel::kv::cache::sm kv_cache{};

  const int32_t ubatch_size = 3;
  CHECK(prepare_kv(kv_cache, &ubatch_size, 1, 16));

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  bool rollback_attempted = false;
  int32_t error = EMEL_OK;
  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = ubatch_size,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .outputs_produced_out = &outputs_produced,
    .kv_tokens_out = &kv_tokens,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error,
  };
  apply_compute_callbacks(execute);
  CHECK(machine.process_event(execute));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(error == EMEL_OK);
  CHECK(machine.outputs_produced() == ubatch_size);
  CHECK(machine.kv_tokens() >= ubatch_size);
  CHECK(outputs_produced == ubatch_size);
  CHECK(kv_tokens >= ubatch_size);
  CHECK_FALSE(rollback_attempted);
}

TEST_CASE("ubatch_executor_execute_rejects_invalid_payload") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t error = EMEL_OK;

  emel::decoder::ubatch_executor::event::execute missing_ubatch{
    .ubatch_index = -1,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .error_out = &error,
  };
  apply_compute_callbacks(missing_ubatch);
  CHECK_FALSE(machine.process_event(missing_ubatch));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  emel::decoder::ubatch_executor::event::execute zero_size{
    .ubatch_index = 0,
    .ubatch_size = 0,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .error_out = &error,
  };
  apply_compute_callbacks(zero_size);
  CHECK_FALSE(machine.process_event(zero_size));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  emel::decoder::ubatch_executor::event::execute missing_memory{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = nullptr,
    .kv_cache_sm = &kv_cache,
    .error_out = &error,
  };
  apply_compute_callbacks(missing_memory);
  CHECK_FALSE(machine.process_event(missing_memory));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_compute_failure_attempts_rollback") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  bool rollback_attempted = false;
  int32_t error = EMEL_OK;

  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error,
  };
  apply_compute_callbacks(execute);
  CHECK_FALSE(machine.process_event(execute));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(error == EMEL_ERR_BACKEND);
  CHECK(rollback_attempted);
}

TEST_CASE("ubatch_executor_rejects_reentrant_execute_when_not_initialized") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  int32_t error_out = EMEL_OK;
  bool rollback_attempted = false;
  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error_out,
  };
  apply_compute_callbacks(execute);
  CHECK_FALSE(machine.process_event(execute));
  CHECK(error_out != EMEL_OK);

  error_out = EMEL_OK;
  rollback_attempted = false;
  emel::decoder::ubatch_executor::event::execute again = execute;
  again.error_out = &error_out;
  again.rollback_attempted_out = &rollback_attempted;
  CHECK_FALSE(machine.process_event(again));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("ubatch_executor_action_helpers_cover_error_branches") {
  using emel::decoder::ubatch_executor::action::context;
  using emel::decoder::ubatch_executor::action::prepare_status_is_error;

  using memory_status = emel::memory::coordinator::event::memory_status;
  CHECK_FALSE(prepare_status_is_error(memory_status::success));
  CHECK_FALSE(prepare_status_is_error(memory_status::no_update));
  CHECK(prepare_status_is_error(memory_status::failed_prepare));
  CHECK(prepare_status_is_error(memory_status::failed_compute));
  CHECK(prepare_status_is_error(static_cast<memory_status>(99)));

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::memory::coordinator::sm memory_coordinator{};
    emel::kv::cache::sm kv_cache{};
    ctx.ubatch_index = -1;
    ctx.ubatch_size = 1;
    emel::decoder::ubatch_executor::event::execute request{
      .ubatch_index = 0,
      .ubatch_size = 1,
      .memory_coordinator_sm = &memory_coordinator,
      .kv_cache_sm = &kv_cache,
    };
    emel::decoder::ubatch_executor::event::validate validate{
      .request = &request,
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_execute_request{}(validate, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_validate(validate, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::decoder::ubatch_executor::event::prepare_memory prepare{
      .memory_coordinator_sm = nullptr,
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_prepare_memory_request{}(prepare, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_prepare_memory(prepare, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::decoder::ubatch_executor::event::prepare_kv prepare{
      .kv_cache_sm = nullptr,
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_prepare_kv_request{}(prepare, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_prepare_kv(prepare, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.outputs_produced = 0;
    emel::decoder::ubatch_executor::event::extract_outputs extract{
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_extract_outputs_request{}(extract, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_extract_outputs(extract, ctx);
    CHECK(error_out != EMEL_OK);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::memory::coordinator::sm memory_coordinator{};
    using mem_base_t = emel::memory::coordinator::sm::base_type;
    auto & mem_base = static_cast<mem_base_t &>(memory_coordinator);
    CHECK(mem_base.process_event(emel::memory::coordinator::event::prepare_batch{
      .n_ubatch = 1,
      .n_ubatches_total = 1,
    }));

    emel::decoder::ubatch_executor::action::run_prepare_memory(
        emel::decoder::ubatch_executor::event::prepare_memory{
            .memory_coordinator_sm = &memory_coordinator,
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::decoder::ubatch_executor::event::run_compute compute{
      .kv_cache_sm = nullptr,
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_run_compute_request{}(compute, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_run_compute(compute, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    int32_t ubatch_size = 1;
    emel::kv::cache::sm kv_cache{};
    CHECK(prepare_kv(kv_cache, &ubatch_size, 1, 8));
    ctx.ubatch_index = 0;
    ctx.ubatch_size = ubatch_size;
    using compute_base_t = emel::decoder::compute_executor::sm::base_type;
    auto & compute_base = static_cast<compute_base_t &>(ctx.compute_executor);
    emel::decoder::compute_executor::event::execute compute_execute{
      .ubatch_index = 0,
      .ubatch_size = 1,
      .kv_tokens = 1,
    };
    apply_compute_callbacks(compute_execute);
    CHECK(compute_base.process_event(compute_execute));

    emel::decoder::ubatch_executor::event::execute compute_request{
      .ubatch_index = 0,
      .ubatch_size = ubatch_size,
      .kv_cache_sm = &kv_cache,
    };
    apply_compute_callbacks(compute_request);
    emel::decoder::ubatch_executor::action::run_compute(
        emel::decoder::ubatch_executor::event::run_compute{
            .kv_cache_sm = &kv_cache,
            .request = &compute_request,
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    emel::decoder::ubatch_executor::event::rollback rollback{
      .kv_cache_sm = nullptr,
      .error_out = &error_out,
    };
    CHECK(emel::decoder::ubatch_executor::guard::invalid_rollback_request{}(rollback, ctx));
    emel::decoder::ubatch_executor::action::reject_invalid_rollback(rollback, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("ubatch_executor_execute_handles_compute_executor_extract_failure") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  bool rollback_attempted = false;
  int32_t error = EMEL_OK;

  const int32_t prepared_size = 1;
  CHECK(prepare_kv(kv_cache, &prepared_size, 1, 1));

  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error,
  };
  apply_compute_callbacks(execute);
  CHECK_FALSE(machine.process_event(execute));
  CHECK(error == EMEL_ERR_BACKEND);
  CHECK(rollback_attempted);
}
