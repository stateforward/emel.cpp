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

TEST_CASE("ubatch_executor_guard_validates_execute_request") {
  emel::decoder::ubatch_executor::action::context ctx{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
  };
  CHECK(emel::decoder::ubatch_executor::guard::valid_execute_request{}(request, ctx));

  request.ubatch_size = 0;
  CHECK(emel::decoder::ubatch_executor::guard::invalid_execute_request{}(request, ctx));
}

TEST_CASE("ubatch_executor_guard_rejects_missing_dependencies") {
  emel::decoder::ubatch_executor::action::context ctx{};
  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = nullptr,
    .kv_cache_sm = nullptr,
  };

  CHECK_FALSE(emel::decoder::ubatch_executor::guard::valid_execute_request{}(request, ctx));
}

TEST_CASE("ubatch_executor_prepare_status_is_error_branches") {
  using emel::decoder::ubatch_executor::action::prepare_status_is_error;
  using emel::memory::coordinator::event::memory_status;

  CHECK_FALSE(prepare_status_is_error(memory_status::success));
  CHECK_FALSE(prepare_status_is_error(memory_status::no_update));
  CHECK(prepare_status_is_error(memory_status::failed_compute));
  CHECK(prepare_status_is_error(static_cast<memory_status>(99)));
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

TEST_CASE("ubatch_executor_run_rollback_keeps_ok_status") {
  emel::decoder::ubatch_executor::action::context ctx{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;

  ctx.ubatch_index = 0;
  emel::decoder::ubatch_executor::action::run_rollback(
    emel::decoder::ubatch_executor::event::rollback{
      .kv_cache_sm = &kv_cache,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("ubatch_executor_action_markers_set_errors") {
  emel::decoder::ubatch_executor::action::context ctx{};

  emel::decoder::ubatch_executor::action::mark_missing_outputs(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK(ctx.execution_error == EMEL_ERR_BACKEND);

  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::ubatch_executor::action::capture_rollback_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.execution_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::ubatch_executor::action::capture_execution_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::ubatch_executor::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("ubatch_executor_run_extract_outputs_phase_keeps_error_clear") {
  emel::decoder::ubatch_executor::action::context ctx{};
  ctx.phase_error = EMEL_ERR_BACKEND;

  emel::decoder::ubatch_executor::action::run_extract_outputs_phase(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("ubatch_executor_on_unexpected_sets_error") {
  emel::decoder::ubatch_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::ubatch_executor::action::on_unexpected{}(
    emel::decoder::ubatch_executor::event::execute{.error_out = &err},
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}
