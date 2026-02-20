#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/recurrent/actions.hpp"
#include "emel/memory/coordinator/recurrent/guards.hpp"
#include "emel/memory/coordinator/recurrent/sm.hpp"

TEST_CASE("memory_coordinator_starts_initialized") {
  emel::memory::coordinator::recurrent::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::recurrent::initialized>));
}

TEST_CASE("memory_coordinator_prepare_update_without_pending_work_returns_no_update") {
  emel::memory::coordinator::recurrent::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
  }));

  CHECK(status == emel::memory::coordinator::event::memory_status::no_update);
}

TEST_CASE("memory_coordinator_prepare_update_without_output_pointer_is_valid") {
  emel::memory::coordinator::recurrent::sm machine{};

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = true,
  }));
}

TEST_CASE("memory_coordinator_prepare_batch_reports_invalid_arguments") {
  emel::memory::coordinator::recurrent::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 0,
    .n_ubatches_total = 2,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 4,
    .n_ubatches_total = 0,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::recurrent::initialized>));
}

TEST_CASE("memory_coordinator_prepare_full_success") {
  emel::memory::coordinator::recurrent::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_compute;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_full{
    .status_out = &status,
  }));
  CHECK(status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_prepare_batch_then_update_applies_pending_work") {
  emel::memory::coordinator::recurrent::sm machine{};

  emel::memory::coordinator::event::memory_status batch_status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 2,
    .n_ubatches_total = 3,
    .status_out = &batch_status,
  }));
  CHECK(batch_status == emel::memory::coordinator::event::memory_status::success);

  emel::memory::coordinator::event::memory_status update_status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &update_status,
  }));
  CHECK(update_status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_guard_helpers_cover_phase_and_apply") {
  using emel::memory::coordinator::recurrent::action::context;
  using emel::memory::coordinator::recurrent::action::request_kind;

  context ctx{};
  CHECK(emel::memory::coordinator::recurrent::guard::apply_update_invalid_context{}(ctx));

  ctx.active_request = request_kind::update;
  ctx.prepared_status = emel::memory::coordinator::event::memory_status::success;
  CHECK_FALSE(emel::memory::coordinator::recurrent::guard::apply_update_invalid_context{}(ctx));

  ctx.has_pending_update = true;
  CHECK(emel::memory::coordinator::recurrent::guard::apply_update_ready{}(ctx));
  CHECK_FALSE(emel::memory::coordinator::recurrent::guard::apply_update_backend_failed{}(ctx));

  ctx.has_pending_update = false;
  ctx.update_request.optimize = true;
  CHECK(emel::memory::coordinator::recurrent::guard::apply_update_ready{}(ctx));

  ctx.update_request.optimize = false;
  CHECK(emel::memory::coordinator::recurrent::guard::apply_update_backend_failed{}(ctx));
}

TEST_CASE("memory_coordinator_wrapper_rejects_new_request_when_not_initialized") {
  emel::memory::coordinator::recurrent::sm machine{};

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
  }));
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::recurrent::initialized>));

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
  }));
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::recurrent::initialized>));
}
