#include <array>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/sequential/actions.hpp"
#include "emel/batch/planner/modes/sequential/guards.hpp"

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

inline emel::batch::planner::event::plan_runtime make_runtime(
    const emel::batch::planner::event::plan_request & request,
    emel::batch::planner::event::plan_scratch & request_ctx) {
  return emel::batch::planner::event::plan_runtime{
    .request = request,
    .ctx = request_ctx,
  };
}

}  // namespace

TEST_CASE("batch_planner_modes_sequential_create_plan_with_masks") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{3U, 1U, 2U, 1U}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::seq,
    .seq_masks = masks.data(),
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 3;

  auto runtime = make_runtime(request, request_ctx);
  emel::batch::planner::modes::sequential::action::effect_begin_planning(runtime, planner_ctx);
  REQUIRE(emel::batch::planner::modes::sequential::guard::guard_sequential_plan_capacity_ok(runtime,
                                                                                       planner_ctx));
  emel::batch::planner::modes::sequential::action::effect_plan_sequential_batches(
      runtime, planner_ctx);
  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 3);
  CHECK(request_ctx.step_sizes[1] == 1);
}

TEST_CASE("batch_planner_modes_sequential_create_plan_without_masks_failure") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::seq,
    .seq_masks = nullptr,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 0;

  auto runtime = make_runtime(request, request_ctx);
  emel::batch::planner::modes::sequential::action::effect_begin_planning(runtime, planner_ctx);
  REQUIRE(emel::batch::planner::modes::sequential::guard::guard_has_invalid_step_size(runtime,
                                                                                 planner_ctx));
  emel::batch::planner::modes::sequential::action::effect_reject_invalid_step_size(
      runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_step_size));
}

TEST_CASE("batch_planner_modes_sequential_marks_progress_stalled") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{42}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::seq,
    .seq_masks = nullptr,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  auto runtime = make_runtime(request, request_ctx);
  emel::batch::planner::modes::sequential::action::effect_begin_planning(runtime, planner_ctx);
  emel::batch::planner::modes::sequential::action::effect_reject_planning_progress_stalled(
      runtime, planner_ctx);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
}
