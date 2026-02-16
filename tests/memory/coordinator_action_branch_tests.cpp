#include <doctest/doctest.h>

#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/events.hpp"
#include "emel/emel.h"

TEST_CASE("memory_coordinator_begin_actions_handle_null_error_out") {
  emel::memory::coordinator::action::context ctx{};

  emel::memory::coordinator::action::begin_prepare_update(
    emel::memory::coordinator::event::prepare_update{}, ctx);
  emel::memory::coordinator::action::begin_prepare_batch(
    emel::memory::coordinator::event::prepare_batch{}, ctx);
  emel::memory::coordinator::action::begin_prepare_full(
    emel::memory::coordinator::event::prepare_full{}, ctx);
}

TEST_CASE("memory_coordinator_validate_actions_check_requests") {
  emel::memory::coordinator::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::memory::coordinator::action::run_validate_update(
    emel::memory::coordinator::event::validate_update{.request = nullptr, .error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::memory::coordinator::action::run_validate_batch(
    emel::memory::coordinator::event::validate_batch{.request = nullptr, .error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::event::prepare_batch batch_req{
    .n_ubatch = 0,
    .n_ubatches_total = 0,
  };
  err = EMEL_OK;
  emel::memory::coordinator::action::run_validate_batch(
    emel::memory::coordinator::event::validate_batch{.request = &batch_req, .error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("memory_coordinator_prepare_steps_handle_invalid_inputs") {
  emel::memory::coordinator::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::memory::coordinator::event::memory_status status = emel::memory::coordinator::event::memory_status::success;

  emel::memory::coordinator::action::run_prepare_update_step(
    emel::memory::coordinator::event::prepare_update_step{
      .request = nullptr,
      .prepared_status_out = &status,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::memory::coordinator::action::run_prepare_batch_step(
    emel::memory::coordinator::event::prepare_batch_step{
      .request = nullptr,
      .prepared_status_out = &status,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("memory_coordinator_apply_update_step_handles_pending_update") {
  emel::memory::coordinator::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_update request{};
  emel::memory::coordinator::action::run_apply_update_step(
    emel::memory::coordinator::event::apply_update_step{
      .request = &request,
      .prepared_status = emel::memory::coordinator::event::memory_status::success,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.has_pending_update = true;
  err = EMEL_OK;
  emel::memory::coordinator::action::run_apply_update_step(
    emel::memory::coordinator::event::apply_update_step{
      .request = &request,
      .prepared_status = emel::memory::coordinator::event::memory_status::success,
      .error_out = &err,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK_FALSE(ctx.has_pending_update);
}
