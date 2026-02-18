#include <doctest/doctest.h>

#include "emel/decoder/compute_executor/actions.hpp"
#include "emel/decoder/compute_executor/events.hpp"
#include "emel/decoder/compute_executor/guards.hpp"

namespace {

using execute_t = emel::decoder::compute_executor::event::execute;

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool prepare_graph_reuse(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  return true;
}

bool alloc_graph_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool bind_inputs_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool run_backend_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool extract_outputs_ok(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool validate_fail(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

bool validate_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_BACKEND;
  }
  return true;
}

bool prepare_graph_fail(const execute_t &, bool *, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

bool alloc_graph_fail(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

bool bind_inputs_fail(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

bool run_backend_fail(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

bool extract_outputs_fail(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 0;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return false;
}

}  // namespace

TEST_CASE("compute_executor_actions_return_early_when_error_out_is_null") {
  emel::decoder::compute_executor::action::context ctx{};
  execute_t request{
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_ok,
    .extract_outputs = extract_outputs_ok,
  };

  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.request = &request}, ctx);
  emel::decoder::compute_executor::action::run_prepare_graph(
    emel::decoder::compute_executor::event::prepare_graph{.request = &request}, ctx);
  emel::decoder::compute_executor::action::run_alloc_graph(
    emel::decoder::compute_executor::event::alloc_graph{.request = &request}, ctx);
  emel::decoder::compute_executor::action::run_bind_inputs(
    emel::decoder::compute_executor::event::bind_inputs{.request = &request}, ctx);
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{.request = &request}, ctx);
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{.request = &request}, ctx);
}

TEST_CASE("compute_executor_guard_validates_execute_request") {
  emel::decoder::compute_executor::action::context ctx{};
  execute_t request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_ok,
    .extract_outputs = extract_outputs_ok,
  };

  CHECK(emel::decoder::compute_executor::guard::valid_execute_request{}(request, ctx));

  request.ubatch_size = 0;
  CHECK(emel::decoder::compute_executor::guard::invalid_execute_request{}(request, ctx));

  request.ubatch_size = 1;
  request.prepare_graph = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_execute_request{}(request, ctx));
}

TEST_CASE("compute_executor_guard_reports_graph_reuse_state") {
  emel::decoder::compute_executor::action::context ctx{};
  ctx.graph_reused = true;
  CHECK(emel::decoder::compute_executor::guard::graph_reused(ctx));
  CHECK_FALSE(emel::decoder::compute_executor::guard::graph_needs_allocation(ctx));

  ctx.graph_reused = false;
  CHECK_FALSE(emel::decoder::compute_executor::guard::graph_reused(ctx));
  CHECK(emel::decoder::compute_executor::guard::graph_needs_allocation(ctx));
}

TEST_CASE("compute_executor_run_validate_handles_missing_and_failed_callbacks") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;
  execute_t request{
    .validate = nullptr,
  };

  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  request.validate = validate_fail;
  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  request.validate = validate_error;
  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_run_phase_actions_report_callback_failures") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;
  bool reused = false;
  execute_t request{
    .prepare_graph = prepare_graph_fail,
    .alloc_graph = alloc_graph_fail,
    .bind_inputs = bind_inputs_fail,
    .run_backend = run_backend_fail,
    .extract_outputs = extract_outputs_fail,
  };

  emel::decoder::compute_executor::action::run_prepare_graph(
    emel::decoder::compute_executor::event::prepare_graph{
      .request = &request,
      .reused_out = &reused,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_alloc_graph(
    emel::decoder::compute_executor::event::alloc_graph{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_bind_inputs(
    emel::decoder::compute_executor::event::bind_inputs{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{.request = &request, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_actions_reject_null_requests") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;
  bool reused = false;

  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.request = nullptr, .error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_prepare_graph(
    emel::decoder::compute_executor::event::prepare_graph{
      .request = nullptr,
      .reused_out = &reused,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_alloc_graph(
    emel::decoder::compute_executor::event::alloc_graph{.request = nullptr, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_bind_inputs(
    emel::decoder::compute_executor::event::bind_inputs{.request = nullptr, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{.request = nullptr, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{.request = nullptr, .error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_actions_handle_missing_callbacks_in_phase_helpers") {
  emel::decoder::compute_executor::action::context ctx{};

  ctx.validate = nullptr;
  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  ctx.prepare_graph = nullptr;
  emel::decoder::compute_executor::action::run_prepare_graph(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.alloc_graph = nullptr;
  emel::decoder::compute_executor::action::run_alloc_graph(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.bind_inputs = nullptr;
  emel::decoder::compute_executor::action::run_bind_inputs(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.run_backend = nullptr;
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.extract_outputs = nullptr;
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_run_extract_outputs_updates_outputs") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;
  execute_t request{
    .extract_outputs = extract_outputs_ok,
  };

  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{
      .request = &request,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.outputs_produced == 1);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("compute_executor_phase_helpers_cover_alloc_graph_path") {
  emel::decoder::compute_executor::action::context ctx{};
  ctx.alloc_graph = alloc_graph_ok;
  ctx.phase_error = EMEL_ERR_BACKEND;

  emel::decoder::compute_executor::action::run_alloc_graph(
    emel::decoder::compute_executor::event::execute{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("compute_executor_on_unexpected_sets_error") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::compute_executor::action::on_unexpected(
    emel::decoder::compute_executor::event::execute{.error_out = &err},
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}
