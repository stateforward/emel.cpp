#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"

namespace {

using compute_execute_t = emel::decoder::compute_executor::event::execute;

bool compute_validate(const compute_execute_t & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.ubatch_index < 0 || ev.ubatch_size <= 0 || ev.kv_tokens < 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
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

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

TEST_CASE("ubatch_executor_sm_rejects_missing_dependencies") {
  emel::decoder::ubatch_executor::sm machine{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = nullptr,
    .kv_cache_sm = nullptr,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);
  machine.process_event(request);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(rollback_attempted);
}

TEST_CASE("ubatch_executor_sm_executes_and_reports_outputs") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;
  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;

  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .outputs_produced_out = &outputs_produced,
    .kv_tokens_out = &kv_tokens,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(execute);
  machine.process_event(execute);
  CHECK(err != EMEL_OK);
  CHECK(rollback_attempted);
}

TEST_CASE("ubatch_executor_sm_validation_error_path") {
  emel::decoder::ubatch_executor::sm machine{};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute execute{
    .ubatch_index = -1,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(execute);
  machine.process_event(execute);
  CHECK(err != EMEL_OK);
  CHECK(rollback_attempted);
}

TEST_CASE("ubatch_executor_testing_policy_prepare_memory_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_prepare_kv_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_run_compute_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::rollback_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_extract_outputs_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::extract_outputs_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::rollback_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_success_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;
  int32_t outputs_produced = 0;
  int32_t kv_tokens = 0;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .outputs_produced_out = &outputs_produced,
    .kv_tokens_out = &kv_tokens,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::extract_outputs_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_done{
    .outputs_produced = 1,
    .kv_tokens = 1,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_validate_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_run_compute_rollback_error_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::rollback_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("ubatch_executor_testing_policy_extract_outputs_rollback_done_path") {
  emel::decoder::ubatch_executor::action::context ctx{};
  noop_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process>
    machine{ctx, process};
  emel::memory::coordinator::sm memory_coordinator{};
  emel::kv::cache::sm kv_cache{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute request{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory_coordinator,
    .kv_cache_sm = &kv_cache,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(request);

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::extract_outputs_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::rollback_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::ubatch_execution_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

}  // namespace
