#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/decoder/sm.hpp"
#include "emel/emel.h"
#include "emel/graph/processor/events.hpp"

namespace {

using decode_event = emel::decoder::event::decode;
using execute_event = emel::graph::processor::event::execute;

bool g_saw_memory_payload = false;

bool compute_validate_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_prepare_graph_expect_memory(const execute_event & ev, bool * reused_out,
                                         int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  g_saw_memory_payload = ev.memory_sm != nullptr &&
                         ev.memory_view.is_sequence_active(0) &&
                         ev.memory_view.lookup_kv_block(0, 0) >= 0 &&
                         ev.memory_view.lookup_recurrent_slot(0) >= 0;
  if (err_out != nullptr) {
    *err_out = g_saw_memory_payload ? EMEL_OK : EMEL_ERR_BACKEND;
  }
  return g_saw_memory_payload;
}

bool compute_prepare_graph_reuse(const execute_event &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_alloc_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_bind_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_run_backend_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_run_backend_fail(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_BACKEND;
  }
  return false;
}

bool compute_extract_outputs_one(const execute_event &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

decode_event make_decode_request() {
  static std::array<int32_t, 1> tokens = {{1}};
  static std::array<int32_t, 1> seq_primary_ids = {{0}};
  static std::array<int32_t, 1> positions = {{0}};

  return decode_event{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
    .output_all = true,
    .seq_primary_ids = seq_primary_ids.data(),
    .seq_primary_ids_count = 1,
    .positions = positions.data(),
    .positions_count = 1,
    .compute_validate = &compute_validate_ok,
    .compute_prepare_graph = &compute_prepare_graph_expect_memory,
    .compute_alloc_graph = &compute_alloc_ok,
    .compute_bind_inputs = &compute_bind_ok,
    .compute_run_backend = &compute_run_backend_ok,
    .compute_extract_outputs = &compute_extract_outputs_one,
    .outputs_capacity = 1,
  };
}

}  // namespace

TEST_CASE("decoder_processes_lifecycle_memory_and_propagates_unified_memory_payload") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;
  g_saw_memory_payload = false;

  decode_event request = make_decode_request();
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(g_saw_memory_payload);
  CHECK(machine.outputs_processed() == 1);
}

TEST_CASE("decoder_propagates_ubatch_failure_and_returns_error") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;

  decode_event request = make_decode_request();
  request.compute_prepare_graph = &compute_prepare_graph_reuse;
  request.compute_run_backend = &compute_run_backend_fail;
  request.error_out = &err;

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}
