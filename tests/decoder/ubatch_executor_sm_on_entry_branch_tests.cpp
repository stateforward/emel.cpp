#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/decoder/ubatch_executor/sm.hpp"
#include "emel/kv/cache/sm.hpp"
#include "emel/memory/coordinator/sm.hpp"
#include "emel/emel.h"

namespace {

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

  CHECK(machine.process_event(exec));
  CHECK(err == EMEL_ERR_BACKEND);

  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::validate_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_memory_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::prepare_kv_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::run_compute_done{.request = &exec}));
  CHECK(machine.process_event(emel::decoder::ubatch_executor::events::extract_outputs_done{.request = &exec}));
}
