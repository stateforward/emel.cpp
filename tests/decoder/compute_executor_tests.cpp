#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/compute_executor/sm.hpp"
#include "emel/emel.h"

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

TEST_CASE("compute_executor_starts_initialized") {
  emel::decoder::compute_executor::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));
}

TEST_CASE("compute_executor_execute_success_path") {
  emel::decoder::compute_executor::sm machine{};
  int32_t outputs_produced = 0;
  int32_t error = EMEL_OK;

  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 1,
    .ubatch_size = 4,
    .kv_tokens = 9,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .outputs_produced_out = &outputs_produced,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(machine.outputs_produced() == 4);
  CHECK(outputs_produced == 4);
}

TEST_CASE("compute_executor_rejects_invalid_payload") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = -1,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 0,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = -1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_handles_backend_and_extract_failures") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .kv_tokens = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_BACKEND);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_runs_to_completion_on_execute") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));

  error = EMEL_OK;
  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 1,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));
}
