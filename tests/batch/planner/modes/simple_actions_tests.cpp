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

}  // namespace

TEST_CASE("batch_planner_modes_simple_create_plan_success") {
  emel::batch::planner::action::context ctx{};
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
  ctx.effective_step_size = 2;

  emel::batch::planner::modes::simple::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 2);
  CHECK(ctx.step_sizes[0] == 2);
  CHECK(ctx.step_sizes[1] == 2);
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_index_overflow") {
  emel::batch::planner::action::context ctx{};
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
  ctx.effective_step_size = static_cast<int32_t>(tokens.size());

  emel::batch::planner::modes::simple::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_step_overflow") {
  emel::batch::planner::action::context ctx{};
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
  ctx.effective_step_size = 1;

  emel::batch::planner::modes::simple::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_planner_modes_simple_create_plan_failure_resets_outputs") {
  emel::batch::planner::action::context ctx{};
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
  ctx.effective_step_size = 0;

  emel::batch::planner::modes::simple::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 0);
  CHECK(ctx.total_outputs == 0);
}
