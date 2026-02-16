#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/sm.hpp"

TEST_CASE("memory_coordinator_starts_initialized") {
  emel::memory::coordinator::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::initialized>));
}

TEST_CASE("memory_coordinator_prepare_update_without_pending_work_returns_no_update") {
  emel::memory::coordinator::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
    .status_out = &status,
  }));

  CHECK(status == emel::memory::coordinator::event::memory_status::no_update);
}

TEST_CASE("memory_coordinator_prepare_update_without_output_pointer_is_valid") {
  emel::memory::coordinator::sm machine{};

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = true,
  }));
}

TEST_CASE("memory_coordinator_prepare_batch_reports_invalid_arguments") {
  emel::memory::coordinator::sm machine{};

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 0,
    .n_ubatches_total = 2,
  }));

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 4,
    .n_ubatches_total = 0,
  }));
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::initialized>));
}

TEST_CASE("memory_coordinator_prepare_full_success") {
  emel::memory::coordinator::sm machine{};
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_compute;

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_full{
    .status_out = &status,
  }));
  CHECK(status == emel::memory::coordinator::event::memory_status::success);
}

TEST_CASE("memory_coordinator_prepare_batch_then_update_applies_pending_work") {
  emel::memory::coordinator::sm machine{};

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

TEST_CASE("memory_coordinator_action_helpers_cover_validation_prepare_apply_publish_edges") {
  using emel::memory::coordinator::action::context;

  context ctx{};
  int32_t error_out = EMEL_OK;

  emel::memory::coordinator::action::run_validate_update(
      emel::memory::coordinator::event::validate_update{
          .request = nullptr,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::run_validate_batch(
      emel::memory::coordinator::event::validate_batch{
          .request = nullptr,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  const emel::memory::coordinator::event::prepare_batch invalid_batch{
    .n_ubatch = 0,
    .n_ubatches_total = 1,
    .status_out = nullptr,
  };
  emel::memory::coordinator::action::run_validate_batch(
      emel::memory::coordinator::event::validate_batch{
          .request = &invalid_batch,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::run_validate_full(
      emel::memory::coordinator::event::validate_full{
          .request = nullptr,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::event::memory_status prepared =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  emel::memory::coordinator::event::prepare_update update_req{
    .optimize = false,
    .status_out = nullptr,
  };
  emel::memory::coordinator::action::run_prepare_update_step(
      emel::memory::coordinator::event::prepare_update_step{
          .request = &update_req,
          .prepared_status_out = nullptr,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::run_prepare_update_step(
      emel::memory::coordinator::event::prepare_update_step{
          .request = &update_req,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(prepared == emel::memory::coordinator::event::memory_status::no_update);

  update_req.optimize = true;
  emel::memory::coordinator::action::run_prepare_update_step(
      emel::memory::coordinator::event::prepare_update_step{
          .request = &update_req,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(prepared == emel::memory::coordinator::event::memory_status::success);

  emel::memory::coordinator::event::prepare_batch batch_req{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
    .status_out = nullptr,
  };
  emel::memory::coordinator::action::run_prepare_batch_step(
      emel::memory::coordinator::event::prepare_batch_step{
          .request = nullptr,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::run_prepare_batch_step(
      emel::memory::coordinator::event::prepare_batch_step{
          .request = &batch_req,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);

  emel::memory::coordinator::event::prepare_full full_req{
    .status_out = nullptr,
  };
  emel::memory::coordinator::action::run_prepare_full_step(
      emel::memory::coordinator::event::prepare_full_step{
          .request = nullptr,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::coordinator::action::run_prepare_full_step(
      emel::memory::coordinator::event::prepare_full_step{
          .request = &full_req,
          .prepared_status_out = &prepared,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);

  emel::memory::coordinator::action::run_apply_update_step(
      emel::memory::coordinator::event::apply_update_step{
          .request = nullptr,
          .prepared_status = emel::memory::coordinator::event::memory_status::success,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  update_req.optimize = false;
  emel::memory::coordinator::action::run_apply_update_step(
      emel::memory::coordinator::event::apply_update_step{
          .request = &update_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.has_pending_update = false;
  emel::memory::coordinator::action::run_apply_update_step(
      emel::memory::coordinator::event::apply_update_step{
          .request = &update_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::success,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  update_req.optimize = true;
  emel::memory::coordinator::action::run_apply_update_step(
      emel::memory::coordinator::event::apply_update_step{
          .request = &update_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::success,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);

  emel::memory::coordinator::event::memory_status published =
      emel::memory::coordinator::event::memory_status::failed_compute;
  update_req.status_out = &published;
  emel::memory::coordinator::action::run_publish_update(
      emel::memory::coordinator::event::publish_update{
          .request = &update_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::no_update,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(published == emel::memory::coordinator::event::memory_status::no_update);

  batch_req.status_out = &published;
  emel::memory::coordinator::action::run_publish_batch(
      emel::memory::coordinator::event::publish_batch{
          .request = &batch_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::success,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(published == emel::memory::coordinator::event::memory_status::success);

  full_req.status_out = &published;
  emel::memory::coordinator::action::run_publish_full(
      emel::memory::coordinator::event::publish_full{
          .request = &full_req,
          .prepared_status = emel::memory::coordinator::event::memory_status::success,
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(published == emel::memory::coordinator::event::memory_status::success);

  emel::memory::coordinator::action::on_memory_done(
      emel::memory::coordinator::events::memory_done{
          .status = emel::memory::coordinator::event::memory_status::success,
      },
      ctx);
  emel::memory::coordinator::action::on_memory_error(
      emel::memory::coordinator::events::memory_error{
          .err = EMEL_ERR_BACKEND,
          .status = emel::memory::coordinator::event::memory_status::failed_prepare,
      },
      ctx);
}

TEST_CASE("memory_coordinator_wrapper_rejects_new_request_when_not_initialized") {
  emel::memory::coordinator::sm machine{};

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
  }));
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::initialized>));

  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
  }));
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::initialized>));
}
