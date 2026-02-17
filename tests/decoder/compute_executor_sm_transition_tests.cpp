#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/decoder/compute_executor/sm.hpp"

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

TEST_CASE("compute_executor_sm_success_path_reports_outputs") {
  emel::decoder::compute_executor::sm machine{};
  int32_t err = EMEL_OK;
  int32_t outputs = 0;

  machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .outputs_produced_out = &outputs,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

TEST_CASE("compute_executor_sm_validation_error_path") {
  emel::decoder::compute_executor::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = -1,
    .ubatch_size = 0,
    .kv_tokens = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
