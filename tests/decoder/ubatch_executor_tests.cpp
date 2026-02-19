#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/emel.h"
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

bool compute_run_backend(const compute_execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = ev.kv_tokens > 0 ? EMEL_OK : EMEL_ERR_BACKEND;
  }
  return ev.kv_tokens > 0;
}

bool compute_extract_outputs(
    const compute_execute_t & ev, int32_t * outputs_out, int32_t * err_out) {
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

void apply_compute_callbacks(emel::decoder::ubatch_executor::event::execute & ev) {
  ev.compute_validate = compute_validate;
  ev.compute_prepare_graph = compute_prepare_graph;
  ev.compute_alloc_graph = compute_alloc_graph;
  ev.compute_bind_inputs = compute_bind_inputs;
  ev.compute_run_backend = compute_run_backend;
  ev.compute_extract_outputs = compute_extract_outputs;
}

bool prepare_kv(
    emel::kv::cache::sm & kv_cache,
    const int32_t * ubatch_sizes,
    const int32_t ubatch_count,
    const int32_t requested_capacity) {
  return kv_cache.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes,
    .ubatch_count = ubatch_count,
    .requested_capacity = requested_capacity,
    .slot_offsets_out = nullptr,
    .slot_offsets_capacity = 0,
    .ubatch_count_out = nullptr,
  });
}

}  // namespace

TEST_CASE("ubatch_executor_starts_initialized") {
  emel::decoder::ubatch_executor::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
}

TEST_CASE("ubatch_executor_execute_success_path") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};

  const int32_t ubatch_size = 3;
  CHECK(prepare_kv(kv_cache, &ubatch_size, 1, 16));

  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;
  bool rollback_attempted = false;
  int32_t error = EMEL_OK;
  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = ubatch_size,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .expected_outputs = ubatch_size,
    .outputs_produced_out = &outputs_produced,
    .kv_tokens_out = &kv_tokens,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error,
  };
  apply_compute_callbacks(execute);
  CHECK(machine.process_event(execute));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(error == EMEL_OK);
  CHECK(outputs_produced == ubatch_size);
  CHECK(kv_tokens >= ubatch_size);
  CHECK_FALSE(rollback_attempted);
}

TEST_CASE("ubatch_executor_execute_rejects_invalid_payload") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t error = EMEL_OK;

  emel::decoder::ubatch_executor::event::execute missing_ubatch{
    .ubatch_index = -1,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .expected_outputs = 2,
    .error_out = &error,
  };
  apply_compute_callbacks(missing_ubatch);
  CHECK_FALSE(machine.process_event(missing_ubatch));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("ubatch_executor_compute_failure_attempts_rollback") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  bool rollback_attempted = false;
  int32_t error = EMEL_OK;

  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .expected_outputs = 2,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &error,
  };
  apply_compute_callbacks(execute);
  CHECK_FALSE(machine.process_event(execute));

  CHECK(machine.is(boost::sml::state<emel::decoder::ubatch_executor::initialized>));
  CHECK(error == EMEL_ERR_BACKEND);
  CHECK(rollback_attempted);
}
