#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/guards.hpp"

TEST_CASE("memory_coordinator_begin_actions_store_requests") {
  emel::memory::coordinator::action::context ctx{};
  int32_t err = EMEL_ERR_BACKEND;
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;

  emel::memory::coordinator::action::begin_prepare_update(
      emel::memory::coordinator::event::prepare_update{
          .optimize = true,
          .status_out = &status,
          .error_out = &err,
      },
      ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.active_request == emel::memory::coordinator::action::request_kind::update);
  CHECK(ctx.update_request.status_out == nullptr);
  CHECK(ctx.update_request.error_out == nullptr);

  emel::memory::coordinator::action::begin_prepare_batch(
      emel::memory::coordinator::event::prepare_batch{
          .n_ubatch = 1,
          .n_ubatches_total = 2,
          .status_out = &status,
          .error_out = &err,
      },
      ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.active_request == emel::memory::coordinator::action::request_kind::batch);
  CHECK(ctx.batch_request.status_out == nullptr);
  CHECK(ctx.batch_request.error_out == nullptr);

  emel::memory::coordinator::action::begin_prepare_full(
      emel::memory::coordinator::event::prepare_full{
          .status_out = &status,
          .error_out = &err,
      },
      ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.active_request == emel::memory::coordinator::action::request_kind::full);
  CHECK(ctx.full_request.status_out == nullptr);
  CHECK(ctx.full_request.error_out == nullptr);
}

TEST_CASE("memory_coordinator_guards_cover_batch_validation") {
  emel::memory::coordinator::action::context ctx{};
  ctx.active_request = emel::memory::coordinator::action::request_kind::batch;
  ctx.batch_request = emel::memory::coordinator::event::prepare_batch{
      .n_ubatch = 0,
      .n_ubatches_total = 1,
  };
  CHECK_FALSE(emel::memory::coordinator::guard::valid_batch_context{}(ctx));
  CHECK(emel::memory::coordinator::guard::invalid_batch_context{}(ctx));

  ctx.batch_request.n_ubatch = 1;
  ctx.batch_request.n_ubatches_total = 2;
  CHECK(emel::memory::coordinator::guard::valid_batch_context{}(ctx));
}

TEST_CASE("memory_coordinator_prepare_update_phase_tracks_status") {
  emel::memory::coordinator::action::context ctx{};
  ctx.active_request = emel::memory::coordinator::action::request_kind::update;

  ctx.has_pending_update = false;
  ctx.update_request.optimize = false;
  emel::memory::coordinator::action::run_prepare_update_phase(ctx);
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::no_update);

  ctx.update_request.optimize = true;
  emel::memory::coordinator::action::run_prepare_update_phase(ctx);
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_apply_update_phase_clears_pending") {
  emel::memory::coordinator::action::context ctx{};
  ctx.active_request = emel::memory::coordinator::action::request_kind::update;
  ctx.has_pending_update = true;
  ctx.update_apply_count = 0;

  emel::memory::coordinator::action::run_apply_update_phase(ctx);
  CHECK_FALSE(ctx.has_pending_update);
  CHECK(ctx.update_apply_count == 1);
}

TEST_CASE("memory_coordinator_error_actions_set_last_error") {
  emel::memory::coordinator::action::context ctx{};

  emel::memory::coordinator::action::set_invalid_argument(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::set_backend_error(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}
