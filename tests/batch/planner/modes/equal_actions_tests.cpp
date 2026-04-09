#include <array>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/equal/actions.hpp"
#include "emel/batch/planner/modes/equal/guards.hpp"
#include "emel/batch/planner/modes/equal/sm.hpp"

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

  void on_done(const emel::batch::planner::modes::equal::events::plan_done &) noexcept {
    calls += 1;
  }
};

struct mode_error_capture {
  int calls = 0;
  emel::error::type err = emel::error::type{};

  void on_error(const emel::batch::planner::modes::equal::events::plan_error & ev) noexcept {
    calls += 1;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::planner::modes::equal::events::plan_done &)>
make_mode_done(mode_done_capture * capture) {
  return emel::callback<void(const emel::batch::planner::modes::equal::events::plan_done &)>::from<
      mode_done_capture,
      &mode_done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::modes::equal::events::plan_error &)>
make_mode_error(mode_error_capture * capture) {
  return emel::callback<void(const emel::batch::planner::modes::equal::events::plan_error &)>::from<
      mode_error_capture,
      &mode_error_capture::on_error>(capture);
}

inline emel::batch::planner::modes::equal::event::plan_runtime make_runtime(
    const emel::batch::planner::event::plan_request & request,
    emel::batch::planner::event::plan_scratch & request_ctx,
    const emel::callback<void(const emel::batch::planner::modes::equal::events::plan_done &)> &
        on_done,
    const emel::callback<void(const emel::batch::planner::modes::equal::events::plan_error &)> &
        on_error) {
  return emel::batch::planner::modes::equal::event::plan_runtime{
    .request = request,
    .ctx = request_ctx,
    .on_done = on_done,
    .on_error = on_error,
  };
}

inline bool run_general_mode_flow(
    const emel::batch::planner::modes::equal::event::plan_runtime & runtime,
    emel::batch::planner::action::context & planner_ctx) {
  (void)planner_ctx;
  emel::batch::planner::modes::equal::sm mode{};
  return mode.process_event(runtime);
}

inline bool run_fast_path_mode_flow(
    const emel::batch::planner::modes::equal::event::plan_runtime & runtime,
    emel::batch::planner::action::context & planner_ctx) {
  (void)planner_ctx;
  emel::batch::planner::modes::equal::sm mode{};
  return mode.process_event(runtime);
}

}  // namespace

TEST_CASE("batch_planner_modes_equal_create_plan_without_masks") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = nullptr,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_general_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 3);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 2);
  CHECK(request_ctx.step_sizes[2] == 1);
  CHECK(mode_done.calls == 1);
  CHECK(mode_error.calls == 0);
}

TEST_CASE("batch_planner_modes_equal_create_plan_skips_nonconsecutive_primary") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<uint64_t, 3> masks = {{1U, 2U, 4U}};
  std::array<int32_t, 3> primary_ids = {{0, 2, 1}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = masks.data(),
    .seq_masks_count = static_cast<int32_t>(masks.size()),
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = true,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_general_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 1);
}

TEST_CASE("batch_planner_modes_equal_create_plan_rejects_zero_batch") {
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
    .mode = emel::batch::planner::event::plan_mode::equal,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 0;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_general_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_step_size));
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_equal_create_plan_fails_when_groups_exceed_capacity") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = masks.data(),
    .seq_masks_count = static_cast<int32_t>(masks.size()),
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_general_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_equal_fast_path_success") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 6> tokens = {{1, 2, 3, 4, 5, 6}};
  std::array<int32_t, 6> primary_ids = {{0, 0, 1, 1, 2, 2}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = true,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 4;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_fast_path_mode_flow(runtime, planner_ctx));

  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 3);
  CHECK(request_ctx.step_sizes[1] == 3);
  CHECK(request_ctx.token_indices_count == 6);
  CHECK(request_ctx.step_token_offsets[0] == 0);
  CHECK(request_ctx.step_token_offsets[1] == 3);
  CHECK(request_ctx.step_token_offsets[2] == 6);
  CHECK(mode_done.calls == 1);
  CHECK(mode_error.calls == 0);
}

TEST_CASE("batch_planner_modes_equal_without_primary_ids_uses_general_path") {
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
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = nullptr,
    .equal_sequential = true,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_fast_path_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 1);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.total_outputs == 1);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::none));
  CHECK(mode_done.calls == 1);
  CHECK(mode_error.calls == 0);
}

TEST_CASE("batch_planner_modes_equal_fast_path_rejects_invalid_sequence_id") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> primary_ids = {{0, -1}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = true,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 2;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_fast_path_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_sequence_id));
}

TEST_CASE("batch_planner_modes_equal_fast_path_stalls_when_step_too_small") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> primary_ids = {{0, 1}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = false,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(run_fast_path_mode_flow(runtime, planner_ctx));
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_equal_fast_path_fails_when_steps_storage_full") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = false,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;
  request_ctx.step_count = emel::batch::planner::action::MAX_PLAN_STEPS;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(emel::batch::planner::modes::equal::guard::guard_lacks_step_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::equal::action::effect_reject_output_steps_full(runtime,
                                                                               planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_steps_full));
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_equal_fast_path_fails_when_indices_storage_full") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = false,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 1;
  request_ctx.token_indices_count = emel::batch::planner::action::MAX_PLAN_STEPS;

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  auto runtime = make_runtime(request, request_ctx, mode_done_cb, mode_error_cb);
  CHECK(emel::batch::planner::modes::equal::guard::guard_lacks_index_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::equal::action::effect_reject_output_indices_full(runtime,
                                                                                 planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.token_indices_count == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_indices_full));
  CHECK(mode_error.calls == 1);
}

TEST_CASE("batch_planner_modes_equal_guards_cover_fast_path_and_decision") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  std::array<uint64_t, 1> masks = {{1U}};
  planner_done_capture done{};
  planner_error_capture error{};
  mode_done_capture mode_done{};
  mode_error_capture mode_error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = nullptr,
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  const auto mode_done_cb = make_mode_done(&mode_done);
  const auto mode_error_cb = make_mode_error(&mode_error);
  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_primary_fast_path(
      make_runtime(request, request_ctx, mode_done_cb, mode_error_cb), planner_ctx));
  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_general_path(
            make_runtime(request, request_ctx, mode_done_cb, mode_error_cb),
            planner_ctx) == false);

  request_ctx.step_count = 1;
  request_ctx.total_outputs = 1;
  request_ctx.token_indices_count = 1;
  request_ctx.step_token_offsets[1] = 1;
  CHECK(emel::batch::planner::modes::equal::guard::guard_planning_succeeded(
      make_runtime(request, request_ctx, mode_done_cb, mode_error_cb),
      planner_ctx));

  request.seq_masks = masks.data();
  request.seq_masks_count = static_cast<int32_t>(masks.size());
  CHECK_FALSE(emel::batch::planner::modes::equal::guard::guard_mode_is_primary_fast_path(
      make_runtime(request, request_ctx, mode_done_cb, mode_error_cb),
      planner_ctx));
  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_general_path(
      make_runtime(request, request_ctx, mode_done_cb, mode_error_cb),
      planner_ctx));

  request_ctx.token_indices_count = 0;
  CHECK(emel::batch::planner::modes::equal::guard::guard_planning_failed(
      make_runtime(request, request_ctx, mode_done_cb, mode_error_cb),
      planner_ctx));
}
