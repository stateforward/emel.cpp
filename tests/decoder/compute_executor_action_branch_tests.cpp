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

bool run_backend_kv_gate(const execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = ev.kv_tokens > 0 ? EMEL_OK : EMEL_ERR_BACKEND;
  }
  return ev.kv_tokens > 0;
}

bool extract_outputs_kv_gate(const execute_t & ev, int32_t * outputs_out, int32_t * err_out) {
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

}  // namespace

TEST_CASE("compute_executor_actions_return_early_when_error_out_is_null") {
  emel::decoder::compute_executor::action::context ctx{};
  execute_t request{
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
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

TEST_CASE("compute_executor_run_validate_reports_invalid_values") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  execute_t request{
    .ubatch_index = -1,
    .ubatch_size = 0,
    .kv_tokens = -1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
  };
  emel::decoder::compute_executor::event::validate validate{
    .request = &request,
    .error_out = &err,
  };
  CHECK_FALSE(emel::decoder::compute_executor::guard::valid_execute_request{}(validate, ctx));
  emel::decoder::compute_executor::action::reject_invalid_validate(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_guards_validate_requests") {
  emel::decoder::compute_executor::action::context ctx{};
  execute_t request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
  };

  emel::decoder::compute_executor::event::validate validate{
    .request = &request,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_execute_request{}(validate, ctx));
  request.ubatch_size = 0;
  CHECK(emel::decoder::compute_executor::guard::invalid_execute_request{}(validate, ctx));
  request.ubatch_size = 1;
  request.prepare_graph = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_execute_request{}(validate, ctx));
  request.prepare_graph = prepare_graph_reuse;

  bool reused = false;
  emel::decoder::compute_executor::event::prepare_graph prepare{
    .request = &request,
    .reused_out = &reused,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_prepare_graph_request{}(prepare, ctx));
  prepare.reused_out = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_prepare_graph_request{}(prepare, ctx));
  prepare.reused_out = &reused;
  request.prepare_graph = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_prepare_graph_request{}(prepare, ctx));
  request.prepare_graph = prepare_graph_reuse;

  emel::decoder::compute_executor::event::alloc_graph alloc{
    .request = &request,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_alloc_graph_request{}(alloc, ctx));
  request.alloc_graph = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_alloc_graph_request{}(alloc, ctx));
  request.alloc_graph = alloc_graph_ok;

  emel::decoder::compute_executor::event::bind_inputs bind{
    .request = &request,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_bind_inputs_request{}(bind, ctx));
  request.bind_inputs = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_bind_inputs_request{}(bind, ctx));
  request.bind_inputs = bind_inputs_ok;

  emel::decoder::compute_executor::event::run_backend run{
    .request = &request,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_run_backend_request{}(run, ctx));
  request.run_backend = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_run_backend_request{}(run, ctx));
  request.run_backend = run_backend_kv_gate;

  emel::decoder::compute_executor::event::extract_outputs extract{
    .request = &request,
  };
  CHECK(emel::decoder::compute_executor::guard::valid_extract_outputs_request{}(extract, ctx));
  request.extract_outputs = nullptr;
  CHECK(emel::decoder::compute_executor::guard::invalid_extract_outputs_request{}(extract, ctx));
}

TEST_CASE("compute_executor_run_backend_requires_tokens") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  execute_t request{
    .kv_tokens = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
  };
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{
      .request = &request,
      .error_out = &err,
    }, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_run_extract_outputs_checks_kv_tokens") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  execute_t request{
    .ubatch_size = 2,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
  };
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{
      .request = &request,
      .error_out = &err,
    }, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
