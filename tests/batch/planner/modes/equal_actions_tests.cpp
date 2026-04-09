#include <array>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/equal/actions.hpp"
#include "emel/batch/planner/modes/equal/guards.hpp"

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

struct mode_done_capture {
  void on_done(const emel::batch::planner::modes::equal::events::plan_done &) noexcept {}
};

struct mode_error_capture {
  void on_error(const emel::batch::planner::modes::equal::events::plan_error &) noexcept {}
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

struct runtime_bundle {
  emel::callback<void(const emel::batch::planner::modes::equal::events::plan_done &)> on_done;
  emel::callback<void(const emel::batch::planner::modes::equal::events::plan_error &)> on_error;
  emel::batch::planner::modes::equal::event::plan_runtime runtime;
  mode_done_capture mode_done = {};
  mode_error_capture mode_error = {};

  runtime_bundle(const emel::batch::planner::event::plan_request & request,
                 emel::batch::planner::event::plan_scratch & request_ctx,
                 done_capture *,
                 error_capture *)
      : on_done(make_mode_done(&mode_done)),
        on_error(make_mode_error(&mode_error)),
        runtime{
            .request = request,
            .ctx = request_ctx,
            .on_done = on_done,
            .on_error = on_error,
        } {
  }

  operator const emel::batch::planner::modes::equal::event::plan_runtime &() const noexcept {
    return runtime;
  }
};

inline runtime_bundle make_runtime(const emel::batch::planner::event::plan_request & request,
                                   emel::batch::planner::event::plan_scratch & request_ctx,
                                   done_capture * done,
                                   error_capture * error) {
  return runtime_bundle{request, request_ctx, done, error};
}

inline void run_general_mode_flow(
    const emel::batch::planner::modes::equal::event::plan_runtime & runtime,
                                  emel::batch::planner::action::context & planner_ctx) {
  using namespace emel::batch::planner::modes::equal;
  action::effect_begin_planning(runtime, planner_ctx);

  if (guard::guard_has_invalid_step_size(runtime, planner_ctx)) {
    action::effect_reject_invalid_step_size(runtime, planner_ctx);
    return;
  }
  if (guard::guard_lacks_step_capacity(runtime, planner_ctx)) {
    action::effect_reject_output_steps_full(runtime, planner_ctx);
    return;
  }
  if (guard::guard_lacks_index_capacity(runtime, planner_ctx)) {
    action::effect_reject_output_indices_full(runtime, planner_ctx);
    return;
  }
  action::effect_plan_equal_batches(runtime, planner_ctx);
}

inline void run_fast_path_mode_flow(
    const emel::batch::planner::modes::equal::event::plan_runtime & runtime,
                                    emel::batch::planner::action::context & planner_ctx) {
  using namespace emel::batch::planner::modes::equal;
  action::effect_begin_planning(runtime, planner_ctx);

  if (guard::guard_has_invalid_step_size(runtime, planner_ctx)) {
    action::effect_reject_invalid_step_size(runtime, planner_ctx);
    return;
  }
  if (guard::guard_fast_path_missing_primary_ids(runtime, planner_ctx)) {
    action::effect_reject_invalid_sequence_id(runtime, planner_ctx);
    return;
  }
  if (guard::guard_fast_path_primary_ids_invalid(runtime, planner_ctx)) {
    action::effect_reject_invalid_sequence_id(runtime, planner_ctx);
    return;
  }
  if (guard::guard_lacks_step_capacity(runtime, planner_ctx)) {
    action::effect_reject_output_steps_full(runtime, planner_ctx);
    return;
  }
  if (guard::guard_lacks_index_capacity(runtime, planner_ctx)) {
    action::effect_reject_output_indices_full(runtime, planner_ctx);
    return;
  }
  action::effect_plan_equal_primary_batches(runtime, planner_ctx);
}

}  // namespace

TEST_CASE("batch_planner_modes_equal_create_plan_without_masks") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_general_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 3);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 2);
  CHECK(request_ctx.step_sizes[2] == 1);
}

TEST_CASE("batch_planner_modes_equal_create_plan_skips_nonconsecutive_primary") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<uint64_t, 3> masks = {{1U, 2U, 4U}};
  std::array<int32_t, 3> primary_ids = {{0, 2, 1}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_general_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 2);
  CHECK(request_ctx.step_sizes[1] == 1);
}

TEST_CASE("batch_planner_modes_equal_create_plan_rejects_zero_batch") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  request_ctx.effective_step_size = 0;

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_general_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_step_size));
}

TEST_CASE("batch_planner_modes_equal_create_plan_fails_when_groups_exceed_capacity") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_general_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
}

TEST_CASE("batch_planner_modes_equal_fast_path_success") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 6> tokens = {{1, 2, 3, 4, 5, 6}};
  std::array<int32_t, 6> primary_ids = {{0, 0, 1, 1, 2, 2}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_fast_path_mode_flow(runtime, planner_ctx);

  CHECK(request_ctx.step_count == 2);
  CHECK(request_ctx.step_sizes[0] == 3);
  CHECK(request_ctx.step_sizes[1] == 3);
  CHECK(request_ctx.token_indices_count == 6);
  CHECK(request_ctx.step_token_offsets[0] == 0);
  CHECK(request_ctx.step_token_offsets[1] == 3);
  CHECK(request_ctx.step_token_offsets[2] == 6);
}

TEST_CASE("batch_planner_modes_equal_fast_path_rejects_missing_primary_ids") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_fast_path_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_sequence_id));
}

TEST_CASE("batch_planner_modes_equal_fast_path_rejects_invalid_sequence_id") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> primary_ids = {{0, -1}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_fast_path_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::invalid_sequence_id));
}

TEST_CASE("batch_planner_modes_equal_fast_path_stalls_when_step_too_small") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> primary_ids = {{0, 1}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  run_fast_path_mode_flow(runtime, planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err ==
        emel::error::cast(emel::batch::planner::error::planning_progress_stalled));
}

TEST_CASE("batch_planner_modes_equal_fast_path_fails_when_steps_storage_full") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  CHECK(emel::batch::planner::modes::equal::guard::guard_lacks_step_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::equal::action::effect_reject_output_steps_full(runtime,
                                                                               planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_steps_full));
}

TEST_CASE("batch_planner_modes_equal_fast_path_fails_when_indices_storage_full") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  done_capture done{};
  error_capture error{};

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

  auto runtime = make_runtime(request, request_ctx, &done, &error);
  CHECK(emel::batch::planner::modes::equal::guard::guard_lacks_index_capacity(runtime, planner_ctx));
  emel::batch::planner::modes::equal::action::effect_reject_output_indices_full(runtime,
                                                                                 planner_ctx);
  CHECK(request_ctx.step_count == 0);
  CHECK(request_ctx.total_outputs == 0);
  CHECK(request_ctx.token_indices_count == 0);
  CHECK(request_ctx.err == emel::error::cast(emel::batch::planner::error::output_indices_full));
}

TEST_CASE("batch_planner_modes_equal_guards_cover_fast_path_and_decision") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::plan_scratch request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  std::array<int32_t, 1> primary_ids = {{0}};
  std::array<uint64_t, 1> masks = {{1U}};
  done_capture done{};
  error_capture error{};

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

  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_primary_fast_path(
      make_runtime(request, request_ctx, &done, &error), planner_ctx));
  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_general_path(
            make_runtime(request, request_ctx, &done, &error), planner_ctx) == false);

  request_ctx.step_count = 1;
  request_ctx.total_outputs = 1;
  request_ctx.token_indices_count = 1;
  request_ctx.step_token_offsets[1] = 1;
  CHECK(emel::batch::planner::modes::equal::guard::guard_planning_succeeded(
      make_runtime(request, request_ctx, &done, &error), planner_ctx));

  request.seq_masks = masks.data();
  request.seq_masks_count = static_cast<int32_t>(masks.size());
  CHECK_FALSE(emel::batch::planner::modes::equal::guard::guard_mode_is_primary_fast_path(
      make_runtime(request, request_ctx, &done, &error), planner_ctx));
  CHECK(emel::batch::planner::modes::equal::guard::guard_mode_is_general_path(
      make_runtime(request, request_ctx, &done, &error), planner_ctx));

  request_ctx.token_indices_count = 0;
  CHECK(emel::batch::planner::modes::equal::guard::guard_planning_failed(
      make_runtime(request, request_ctx, &done, &error), planner_ctx));
}
