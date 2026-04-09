#include <array>
#include <vector>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/simple/actions.hpp"
#include "emel/batch/planner/modes/simple/guards.hpp"

namespace {

struct planner_done_capture {
  void on_done(const emel::batch::planner::events::plan_done &) noexcept {}
};

struct planner_error_capture {
  void on_error(const emel::batch::planner::events::plan_error &) noexcept {}
};

inline emel::callback<void(const emel::batch::planner::events::plan_done &)> make_done(
    planner_done_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
      planner_done_capture,
      &planner_done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::events::plan_error &)> make_error(
    planner_error_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
      planner_error_capture,
      &planner_error_capture::on_error>(capture);
}

struct mode_done_capture {
  int calls = 0;

  void on_done(const emel::batch::planner::modes::simple::events::plan_done &) noexcept {
    calls += 1;
  }
};

struct mode_error_capture {
  int calls = 0;
  emel::error::type err = emel::error::type{};

  void on_error(const emel::batch::planner::modes::simple::events::plan_error & ev) noexcept {
    calls += 1;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::planner::modes::simple::events::plan_done &)>
make_mode_done(mode_done_capture * capture) {
  return emel::callback<void(const emel::batch::planner::modes::simple::events::plan_done &)>::from<
      mode_done_capture,
      &mode_done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::modes::simple::events::plan_error &)>
make_mode_error(mode_error_capture * capture) {
  return emel::callback<void(const emel::batch::planner::modes::simple::events::plan_error &)>::from<
      mode_error_capture,
      &mode_error_capture::on_error>(capture);
}

inline emel::batch::planner::modes::simple::event::plan_runtime make_runtime(
    const emel::batch::planner::event::plan_request & request,
    emel::batch::planner::event::plan_scratch & request_ctx,
    const emel::callback<void(const emel::batch::planner::modes::simple::events::plan_done &)> &
        on_done,
    const emel::callback<void(const emel::batch::planner::modes::simple::events::plan_error &)> &
        on_error) {
  return emel::batch::planner::modes::simple::event::plan_runtime{
    .request = request,
    .ctx = request_ctx,
    .on_done = on_done,
    .on_error = on_error,
  };
}

}  // namespace

TEST_CASE("batch_planner_modes_simple_create_plan_success") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  const auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  emel::batch::planner::modes::simple::action::effect_begin_planning(runtime, planner_ctx);
  CHECK(emel::batch::planner::modes::simple::guard::guard_simple_plan_capacity_ok(runtime, planner_ctx));
  emel::batch::planner::modes::simple::action::effect_plan_simple_batches(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 2);
  emel::batch::planner::modes::simple::action::effect_emit_plan_done(runtime, planner_ctx);
  CHECK(mode_done.calls == 1);
  CHECK(mode_error.calls == 0);
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_index_overflow") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::planner::action::MAX_PLAN_STEPS) + 1U;
  std::vector<int32_t> tokens(token_count, 1);
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = static_cast<int32_t>(tokens.size());

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  const auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  emel::batch::planner::modes::simple::action::effect_begin_planning(runtime, planner_ctx);
  CHECK(emel::batch::planner::modes::simple::guard::guard_exceeds_index_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::simple::action::effect_reject_output_indices_full(runtime,
                                                                                  planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_indices_full));
  CHECK(mode_done.calls == 0);
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_simple_create_plan_fails_on_step_overflow") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::planner::action::MAX_PLAN_STEPS) + 1U;
  std::vector<int32_t> tokens(token_count, 1);
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  const auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  emel::batch::planner::modes::simple::action::effect_begin_planning(runtime, planner_ctx);
  CHECK(emel::batch::planner::modes::simple::guard::guard_exceeds_step_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::simple::action::effect_reject_output_steps_full(runtime,
                                                                                planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_steps_full));
  CHECK(mode_done.calls == 0);
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_simple_create_plan_failure_resets_outputs") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 0;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  const auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  emel::batch::planner::modes::simple::action::effect_begin_planning(runtime, planner_ctx);
  CHECK(emel::batch::planner::modes::simple::guard::guard_has_invalid_step_size(runtime, planner_ctx));
  emel::batch::planner::modes::simple::action::effect_reject_invalid_step_size(runtime,
                                                                                planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_step_size));
  CHECK(mode_done.calls == 0);
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_simple_marks_progress_stalled") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{42}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  const auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  emel::batch::planner::modes::simple::action::effect_begin_planning(runtime, planner_ctx);
  emel::batch::planner::modes::simple::action::effect_reject_planning_progress_stalled(
      runtime, planner_ctx);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
  CHECK(mode_done.calls == 0);
  CHECK(mode_error.calls == 1);
}
