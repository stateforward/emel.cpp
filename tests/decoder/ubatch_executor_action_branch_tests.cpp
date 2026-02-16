#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

TEST_CASE("ubatch_executor_actions_return_early_when_error_out_is_null") {
  emel::decoder::ubatch_executor::action::context ctx{};

  emel::decoder::ubatch_executor::action::run_validate(
    emel::decoder::ubatch_executor::event::validate{}, ctx);
  emel::decoder::ubatch_executor::action::run_prepare_memory(
    emel::decoder::ubatch_executor::event::prepare_memory{}, ctx);
  emel::decoder::ubatch_executor::action::run_prepare_kv(
    emel::decoder::ubatch_executor::event::prepare_kv{}, ctx);
  emel::decoder::ubatch_executor::action::run_compute(
    emel::decoder::ubatch_executor::event::run_compute{}, ctx);
  emel::decoder::ubatch_executor::action::run_extract_outputs(
    emel::decoder::ubatch_executor::event::extract_outputs{}, ctx);
  emel::decoder::ubatch_executor::action::run_rollback(
    emel::decoder::ubatch_executor::event::rollback{}, ctx);
}

TEST_CASE("ubatch_executor_run_validate_checks_indices") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.ubatch_index = -1;
  ctx.ubatch_size = 0;
  emel::decoder::ubatch_executor::action::run_validate(
    emel::decoder::ubatch_executor::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_prepare_memory_requires_memory_coordinator") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::action::run_prepare_memory(
    emel::decoder::ubatch_executor::event::prepare_memory{
      .memory_coordinator_sm = nullptr,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_prepare_kv_requires_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::action::run_prepare_kv(
    emel::decoder::ubatch_executor::event::prepare_kv{
      .kv_cache_sm = nullptr,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_compute_reports_missing_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::action::run_compute(
    emel::decoder::ubatch_executor::event::run_compute{
      .kv_cache_sm = nullptr,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_compute_reports_kv_apply_failure") {
  emel::decoder::ubatch_executor::action::context ctx{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;

  ctx.ubatch_index = 0;
  ctx.ubatch_size = 1;
  emel::decoder::ubatch_executor::action::run_compute(
    emel::decoder::ubatch_executor::event::run_compute{
      .kv_cache_sm = &kv_cache,
      .error_out = &err,
    },
    ctx);
  CHECK(err != EMEL_OK);
}

TEST_CASE("ubatch_executor_run_extract_outputs_reports_empty") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_produced = 0;
  emel::decoder::ubatch_executor::action::run_extract_outputs(
    emel::decoder::ubatch_executor::event::extract_outputs{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("ubatch_executor_run_rollback_requires_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::action::run_rollback(
    emel::decoder::ubatch_executor::event::rollback{
      .kv_cache_sm = nullptr,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
