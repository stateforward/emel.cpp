#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/decoder/ubatch_executor/guards.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace {

using compute_execute_t = emel::decoder::compute_executor::event::execute;

bool compute_validate(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
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

bool compute_run_backend(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_extract_outputs(
    const compute_execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
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

}  // namespace

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

  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  };
  ctx.ubatch_index = -1;
  ctx.ubatch_size = 0;
  emel::decoder::ubatch_executor::event::validate validate{
    .request = &request,
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_execute_request{}(validate, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_validate(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_prepare_memory_requires_memory_coordinator") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::event::prepare_memory prepare{
    .memory_coordinator_sm = nullptr,
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_prepare_memory_request{}(prepare, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_prepare_memory(prepare, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_prepare_kv_requires_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::event::prepare_kv prepare{
    .kv_cache_sm = nullptr,
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_prepare_kv_request{}(prepare, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_prepare_kv(prepare, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_compute_reports_missing_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::event::run_compute compute{
    .kv_cache_sm = nullptr,
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_run_compute_request{}(compute, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_run_compute(compute, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_run_compute_reports_kv_apply_failure") {
  emel::decoder::ubatch_executor::action::context ctx{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;

  ctx.ubatch_index = 0;
  ctx.ubatch_size = 1;
  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_cache_sm = &kv_cache,
  };
  apply_compute_callbacks(request);
  emel::decoder::ubatch_executor::action::run_compute(
    emel::decoder::ubatch_executor::event::run_compute{
      .kv_cache_sm = &kv_cache,
      .request = &request,
      .error_out = &err,
    },
    ctx);
  CHECK(err != EMEL_OK);
}

TEST_CASE("ubatch_executor_run_extract_outputs_reports_empty") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_produced = 0;
  emel::decoder::ubatch_executor::event::extract_outputs extract{
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_extract_outputs_request{}(extract, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_extract_outputs(extract, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("ubatch_executor_run_rollback_requires_kv_cache") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::event::rollback rollback{
    .kv_cache_sm = nullptr,
    .error_out = &err,
  };
  CHECK(emel::decoder::ubatch_executor::guard::invalid_rollback_request{}(rollback, ctx));
  emel::decoder::ubatch_executor::action::reject_invalid_rollback(rollback, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
