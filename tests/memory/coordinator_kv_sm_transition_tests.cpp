#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/kv/sm.hpp"

namespace {

TEST_CASE("memory_coordinator_sm_prepare_update_no_update") {
  emel::memory::coordinator::kv::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(status == emel::memory::coordinator::event::memory_status::no_update);
}

TEST_CASE("memory_coordinator_sm_prepare_batch_then_update_success") {
  emel::memory::coordinator::kv::sm machine{};
  emel::memory::coordinator::event::memory_status batch_status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 1,
    .n_ubatches_total = 2,
    .status_out = &batch_status,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(batch_status == emel::memory::coordinator::event::memory_status::success);

  emel::memory::coordinator::event::memory_status update_status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  err = EMEL_OK;
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &update_status,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(update_status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_sm_prepare_full_and_validation_error") {
  emel::memory::coordinator::kv::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_full{
    .status_out = &status,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(status == emel::memory::coordinator::event::memory_status::success);

  err = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 0,
    .n_ubatches_total = 1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

}  // namespace
