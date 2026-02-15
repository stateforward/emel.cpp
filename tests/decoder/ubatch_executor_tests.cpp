#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace {

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
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  const int32_t ubatch_size = 3;
  CHECK(prepare_kv(kv_cache, &ubatch_size, 1, 16));

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  CHECK(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = ubatch_size,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .outputs_produced_out = &outputs_produced,
    .kv_tokens_out = &kv_tokens,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(machine.status_code() == EMEL_OK);
  CHECK(machine.outputs_produced() == ubatch_size);
  CHECK(machine.kv_tokens() >= ubatch_size);
  CHECK(outputs_produced == ubatch_size);
  CHECK(kv_tokens >= ubatch_size);
  CHECK_FALSE(machine.rollback_attempted());
}

TEST_CASE("ubatch_executor_execute_rejects_invalid_payload") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = -1,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));
  CHECK(machine.status_code() == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 0,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));
  CHECK(machine.status_code() == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = nullptr,
    .kv_cache_sm = &kv_cache,
  }));
  CHECK(machine.status_code() == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_compute_failure_attempts_rollback") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(machine.status_code() == EMEL_ERR_BACKEND);
  CHECK(machine.rollback_attempted());
}

TEST_CASE("ubatch_executor_rejects_reentrant_execute_when_not_initialized") {
  emel::decoder::ubatch_executor::sm machine{};
  using base_type = emel::decoder::ubatch_executor::sm::base_type;
  auto & base = static_cast<base_type &>(machine);

  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  CHECK(base.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));
  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::validating>));

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));

  CHECK(base.process_event(emel::decoder::ubatch_executor::events::validate_error{
    .err = EMEL_ERR_BACKEND,
  }));
  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::errored>));

  CHECK(base.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
  }));
  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
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
    ctx.memory_coordinator_sm = &memory_coordinator;
    ctx.kv_cache_sm = &kv_cache;
    emel::decoder::ubatch_executor::action::run_validate(
        emel::decoder::ubatch_executor::event::validate{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.status_code == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.memory_coordinator_sm = nullptr;
    emel::decoder::ubatch_executor::action::run_prepare_memory(
        emel::decoder::ubatch_executor::event::prepare_memory{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.status_code == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.kv_cache_sm = nullptr;
    emel::decoder::ubatch_executor::action::run_prepare_kv(
        emel::decoder::ubatch_executor::event::prepare_kv{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.status_code == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.ubatch_size = 0;
    emel::decoder::ubatch_executor::action::run_extract_outputs(
        emel::decoder::ubatch_executor::event::extract_outputs{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
    CHECK(ctx.status_code == EMEL_ERR_BACKEND);
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

    ctx.memory_coordinator_sm = &memory_coordinator;
    emel::decoder::ubatch_executor::action::run_prepare_memory(
        emel::decoder::ubatch_executor::event::prepare_memory{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
    CHECK(ctx.status_code == EMEL_ERR_BACKEND);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.kv_cache_sm = nullptr;
    emel::decoder::ubatch_executor::action::run_compute(
        emel::decoder::ubatch_executor::event::run_compute{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.status_code == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    int32_t ubatch_size = 1;
    emel::kv::cache::sm kv_cache{};
    CHECK(prepare_kv(kv_cache, &ubatch_size, 1, 8));
    ctx.ubatch_index = 0;
    ctx.ubatch_size = ubatch_size;
    ctx.kv_cache_sm = &kv_cache;
    using compute_base_t = emel::decoder::compute_executor::sm::base_type;
    auto & compute_base = static_cast<compute_base_t &>(ctx.compute_executor);
    CHECK(compute_base.process_event(emel::decoder::compute_executor::event::execute{
      .ubatch_index = 0,
      .ubatch_size = 1,
      .kv_tokens = 1,
    }));

    emel::decoder::ubatch_executor::action::run_compute(
        emel::decoder::ubatch_executor::event::run_compute{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
    CHECK(ctx.status_code == EMEL_ERR_BACKEND);
  }

  {
    context ctx{};
    int32_t error_out = EMEL_OK;
    ctx.kv_cache_sm = nullptr;
    emel::decoder::ubatch_executor::action::run_rollback(
        emel::decoder::ubatch_executor::event::rollback{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.status_code == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.rollback_attempted);
  }
}

TEST_CASE("ubatch_executor_execute_handles_compute_executor_extract_failure") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  const int32_t prepared_size = 1;
  CHECK(prepare_kv(kv_cache, &prepared_size, 1, 16));

  CHECK_FALSE(machine.process_event(emel::decoder::ubatch_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  }));
  CHECK(machine.status_code() == EMEL_ERR_BACKEND);
  CHECK(machine.rollback_attempted());
}
