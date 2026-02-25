#include <array>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/sequential/actions.hpp"

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

TEST_CASE("batch_planner_modes_sequential_create_plan_with_masks") {
  emel::batch::planner::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{3U, 1U, 2U, 1U}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::seq,
    .seq_masks = masks.data(),
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  ctx.effective_step_size = 3;

  emel::batch::planner::modes::sequential::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 2);
  CHECK(ctx.step_sizes[0] == 3);
  CHECK(ctx.step_sizes[1] == 1);
}

TEST_CASE("batch_planner_modes_sequential_create_plan_without_masks_failure") {
  emel::batch::planner::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::seq,
    .seq_masks = nullptr,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  ctx.effective_step_size = 0;

  emel::batch::planner::modes::sequential::action::create_plan(request, ctx);
  CHECK(ctx.step_count == 0);
  CHECK(ctx.total_outputs == 0);
}
