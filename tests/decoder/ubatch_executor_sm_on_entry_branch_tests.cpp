#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"
#include "emel/emel.h"

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

struct error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

}  // namespace

TEST_CASE("ubatch_executor_sm_on_entry_branches_take_error_paths") {
  emel::decoder::ubatch_executor::action::context ctx{};
  error_queue queue{};
  emel::decoder::ubatch_executor::Process process{queue};
  boost::sml::sm<
    emel::decoder::ubatch_executor::model,
    boost::sml::testing,
    emel::decoder::ubatch_executor::Process> machine{ctx, process};

  emel::memory::coordinator::sm memory{};
  emel::kv::cache::sm kv{};
  int32_t err = EMEL_OK;
  bool rollback_attempted = false;

  emel::decoder::ubatch_executor::event::execute exec{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .memory_coordinator_sm = &memory,
    .kv_cache_sm = &kv,
    .rollback_attempted_out = &rollback_attempted,
    .error_out = &err,
  };
  apply_compute_callbacks(exec);

  CHECK(machine.process_event(exec));
  CHECK(err == EMEL_ERR_BACKEND);

  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::extract_outputs_done{.request = &exec}));
}
