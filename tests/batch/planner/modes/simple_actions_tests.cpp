#include <array>
#include <vector>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/simple/actions.hpp"

namespace {

struct done_capture {
  void on_done(const emel::batch::planner::events::plan_done &) noexcept {}
};

struct error_capture {
  void on_error(const emel::batch::planner::events::plan_error &) noexcept {}
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

inline emel::batch::planner::event::request_runtime make_runtime(
    const emel::batch::planner::event::request & request,
    emel::batch::planner::event::request_ctx & request_ctx) {
  return emel::batch::planner::event::request_runtime{
    .request = request,
    .ctx = request_ctx,
  };
}

}  // namespace

TEST_CASE("batch_planner_modes_simple_create_plan_success") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  emel::batch::planner::modes::simple::action::create_plan(
      make_runtime(request, request_ctx), planner_ctx);
  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 2);
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_index_overflow") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::planner::action::MAX_PLAN_STEPS) + 1U;
  std::vector<int32_t> tokens(token_count, 1);
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = static_cast<int32_t>(tokens.size());

  emel::batch::planner::modes::simple::action::create_plan(
      make_runtime(request, request_ctx), planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_indices_full));
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_step_overflow") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::planner::action::MAX_PLAN_STEPS) + 1U;
  std::vector<int32_t> tokens(token_count, 1);
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;

  emel::batch::planner::modes::simple::action::create_plan(
      make_runtime(request, request_ctx), planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_steps_full));
}

TEST_CASE("batch_planner_modes_simple_create_plan_failure_resets_outputs") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 0;

  emel::batch::planner::modes::simple::action::create_plan(
      make_runtime(request, request_ctx), planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_step_size));
}
