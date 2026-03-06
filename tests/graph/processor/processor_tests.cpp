#include <boost/sml.hpp>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/graph/processor/sm.hpp"
#include "emel/memory/view.hpp"

namespace {

using execute_t = emel::graph::processor::event::execute;

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

bool prepare_graph_reuse(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
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

bool run_backend_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

bool run_backend_fail(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::kernel_failed));
  }
  return false;
}

bool extract_outputs_one(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  }
  return true;
}

void * g_expected_memory_sm = nullptr;
bool g_saw_unified_payload = false;

bool prepare_graph_checks_memory_payload(const execute_t & ev, bool * reused_out,
                                         int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  g_saw_unified_payload = ev.memory_sm == g_expected_memory_sm &&
                          ev.memory_view != nullptr &&
                          ev.memory_view->is_sequence_active(7) &&
                          ev.memory_view->sequence_length(7) == 13 &&
                          ev.memory_view->lookup_kv_block(7, 0) == 42 &&
                          ev.memory_view->lookup_recurrent_slot(7) == 3;
  if (err_out != nullptr) {
    *err_out = g_saw_unified_payload ? static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)) : static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::kernel_failed));
  }
  return g_saw_unified_payload;
}

}  // namespace

TEST_CASE("graph_processor_starts_initialized") {
  emel::graph::processor::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::graph::processor::initialized>));
}

TEST_CASE("graph_processor_execute_success_path") {
  emel::graph::processor::sm machine{};
  int32_t outputs_produced = 0;
  int32_t error = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));

  CHECK(machine.process_event(emel::graph::processor::event::execute{
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_ok,
    .extract_outputs = extract_outputs_one,
    .outputs_produced_out = &outputs_produced,
    .error_out = &error,
  }));
  CHECK(error == static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)));
  CHECK(outputs_produced == 1);
  CHECK(machine.outputs_produced() == 1);
}

TEST_CASE("graph_processor_rejects_invalid_payload") {
  emel::graph::processor::sm machine{};
  int32_t error = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));

  CHECK_FALSE(machine.process_event(emel::graph::processor::event::execute{
    .step_index = -1,
    .step_size = 1,
    .kv_tokens = 0,
    .prepare_graph = prepare_graph_reuse,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_ok,
    .extract_outputs = extract_outputs_one,
    .error_out = &error,
  }));
  CHECK(error == static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::invalid_request)));
}

TEST_CASE("graph_processor_propagates_unified_memory_payload") {
  emel::graph::processor::sm machine{};
  int32_t outputs_produced = 0;
  int32_t error = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));
  int32_t memory_tag = 123;
  g_expected_memory_sm = &memory_tag;
  g_saw_unified_payload = false;

  emel::memory::view::snapshot memory_snapshot{};
  memory_snapshot.max_sequences = 8;
  memory_snapshot.block_tokens = 16;
  memory_snapshot.sequence_active[7] = 1;
  memory_snapshot.sequence_length_values[7] = 13;
  memory_snapshot.sequence_kv_block_count[7] = 1;
  memory_snapshot.sequence_kv_blocks[7][0] = 42;
  memory_snapshot.sequence_recurrent_slot[7] = 3;
  CHECK(machine.process_event(emel::graph::processor::event::execute{
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .memory_sm = &memory_tag,
    .memory_view = &memory_snapshot,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_checks_memory_payload,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_ok,
    .extract_outputs = extract_outputs_one,
    .outputs_produced_out = &outputs_produced,
    .error_out = &error,
  }));
  CHECK(error == static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none)));
  CHECK(g_saw_unified_payload);
}

TEST_CASE("graph_processor_propagates_backend_failure") {
  emel::graph::processor::sm machine{};
  int32_t error = static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::none));

  CHECK_FALSE(machine.process_event(emel::graph::processor::event::execute{
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_backend = run_backend_fail,
    .extract_outputs = extract_outputs_one,
    .error_out = &error,
  }));
  CHECK(error == static_cast<int32_t>(emel::error::cast(emel::graph::processor::error::kernel_failed)));
}
