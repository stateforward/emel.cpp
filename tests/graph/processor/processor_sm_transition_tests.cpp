#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/graph/processor/sm.hpp"

// TODO(rearchitecture-cleanup): Keep legacy "compute_executor_*" test names until
// external references to current test IDs are migrated.

namespace {

using execute_t = emel::graph::processor::event::execute;

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

bool prepare_graph_reuse(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  return true;
}

bool alloc_graph_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

bool bind_inputs_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

bool run_backend_kv_gate(const execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = ev.kv_tokens > 0 ? static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)) : static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::kernel_failed));
  }
  return ev.kv_tokens > 0;
}

bool extract_outputs_kv_gate(const execute_t & ev, int32_t * outputs_out, int32_t * err_out) {
  if (ev.kv_tokens < ev.step_size) {
    if (err_out != nullptr) {
      *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::kernel_failed));
    }
    return false;
  }
  if (outputs_out != nullptr) {
    *outputs_out = ev.step_size;
  }
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

TEST_CASE("compute_executor_sm_success_path_reports_outputs") {
  emel::graph::processor::sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  int32_t outputs = 0;

  machine.process_event(emel::graph::processor::event::execute{
    .step_index = 0,
    .step_size = 1,
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
  CHECK(err != static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)));
}

TEST_CASE("compute_executor_sm_validation_error_path") {
  emel::graph::processor::sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));

  machine.process_event(emel::graph::processor::event::execute{
    .step_index = -1,
    .step_size = 0,
    .kv_tokens = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_kv_gate,
    .extract_outputs = extract_outputs_kv_gate,
    .error_out = &err,
  });
  CHECK(err != static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)));
}

}  // namespace
