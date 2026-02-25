#include <array>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/errors.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"


namespace {

struct done_capture {
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  int32_t calls = 0;

  void on_done(const emel::batch::planner::events::plan_done & ev) noexcept {
    calls += 1;
    step_count = ev.step_count;
    total_outputs = ev.total_outputs;
  }
};

struct error_capture {
  int32_t err = EMEL_OK;
  int32_t calls = 0;

  void on_error(const emel::batch::planner::events::plan_error & ev) noexcept {
    calls += 1;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::planner::events::plan_done &)> make_done(
    done_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
    done_capture,
    &done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::events::plan_error &)> make_error(
    error_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
    error_capture,
    &error_capture::on_error>(capture);
}

}  // namespace

TEST_CASE("batch_planner_actions_begin_plan_copies_request") {
  emel::batch::planner::action::context ctx{};
  done_capture done{};
  error_capture error{};
  ctx.effective_step_size = 7;
  ctx.step_count = 2;
  ctx.total_outputs = 9;
  ctx.token_indices_count = 3;

  emel::batch::planner::event::request request{
    .n_tokens = 3,
    .n_steps = 2,
    .mode = emel::batch::planner::event::plan_mode::seq,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  emel::batch::planner::action::begin_plan(request, ctx);

  CHECK(ctx.effective_step_size == 0);
  CHECK(ctx.step_count == 0);
  CHECK(ctx.total_outputs == 0);
  CHECK(ctx.token_indices_count == 0);
}

TEST_CASE("batch_planner_actions_normalize_batch_clamps_requested") {
  emel::batch::planner::action::context ctx{};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .n_tokens = 4,
    .n_steps = 0,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  emel::batch::planner::action::normalize_batch(request, ctx);
  CHECK(ctx.effective_step_size == 4);

  request.n_steps = 10;
  emel::batch::planner::action::normalize_batch(request, ctx);
  CHECK(ctx.effective_step_size == 4);
}

TEST_CASE("batch_planner_actions_dispatch_helpers_cover_callbacks") {
  emel::batch::planner::action::context ctx{};
  done_capture done{};
  error_capture error{};

  ctx.step_sizes[0] = 2;
  ctx.step_count = 1;
  ctx.total_outputs = 2;

  emel::batch::planner::event::request request{
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  emel::batch::planner::action::dispatch_done(request, ctx);
  CHECK(done.calls == 1);
  CHECK(done.step_count == 1);
  CHECK(done.total_outputs == 2);

  emel::batch::planner::action::dispatch_invalid_request(
    request,
    emel::error::cast(emel::batch::planner::error::invalid_request));
  CHECK(error.calls == 1);
  CHECK(error.err == emel::error::cast(emel::batch::planner::error::invalid_request));

  emel::batch::planner::action::dispatch_plan_failed(
    request,
    emel::error::cast(emel::batch::planner::error::internal_error));
  CHECK(error.calls == 2);
  CHECK(error.err == emel::error::cast(emel::batch::planner::error::internal_error));

  emel::batch::planner::action::dispatch_unexpected(request);
  CHECK(error.calls == 3);
  CHECK(error.err ==
        emel::error::cast(emel::batch::planner::error::invalid_request));
}

TEST_CASE("batch_planner_actions_dispatch_helpers_require_callbacks") {
  emel::batch::planner::action::context ctx{};
  done_capture done{};
  error_capture error{};
  auto on_done =
    emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
      done_capture,
      &done_capture::on_done>(&done);
  auto on_error =
    emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
      error_capture,
      &error_capture::on_error>(&error);

  emel::batch::planner::event::request request{
    .on_done = on_done,
    .on_error = on_error,
  };

  emel::batch::planner::action::dispatch_done(request, ctx);
  emel::batch::planner::action::dispatch_invalid_request(
    request,
    emel::error::cast(emel::batch::planner::error::invalid_request));
  emel::batch::planner::action::dispatch_plan_failed(
    request,
    emel::error::cast(emel::batch::planner::error::internal_error));
  emel::batch::planner::action::dispatch_unexpected(request);
  CHECK(done.calls == 1);
  CHECK(error.calls == 3);
}
