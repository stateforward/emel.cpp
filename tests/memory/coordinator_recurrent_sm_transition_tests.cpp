#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/recurrent/sm.hpp"

namespace {

emel::memory::coordinator::event::memory_status prepare_update_success(
    const emel::memory::coordinator::event::prepare_update &,
    void *,
    int32_t * err_out) noexcept {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return emel::memory::coordinator::event::memory_status::success;
}

emel::memory::coordinator::event::memory_status prepare_batch_failed(
    const emel::memory::coordinator::event::prepare_batch &,
    void *,
    int32_t * err_out) noexcept {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return emel::memory::coordinator::event::memory_status::failed_prepare;
}

TEST_CASE("memory_coordinator_sm_prepare_update_no_update") {
  emel::memory::coordinator::recurrent::sm machine{};
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
  emel::memory::coordinator::recurrent::sm machine{};
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
    .prepare_fn = prepare_update_success,
  }));
  CHECK(err == EMEL_OK);
  CHECK(update_status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_sm_prepare_full_and_validation_error") {
  emel::memory::coordinator::recurrent::sm machine{};
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

TEST_CASE("memory_coordinator_sm_prepare_batch_hook_failure") {
  emel::memory::coordinator::recurrent::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
    .error_out = &err,
    .prepare_fn = prepare_batch_failed,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
}

}  // namespace
