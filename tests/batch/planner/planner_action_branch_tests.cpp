#include <array>
#include <doctest/doctest.h>

#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/callback.hpp"

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

TEST_CASE("batch_planner_guard_inputs_are_valid") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = nullptr,
    .n_tokens = 0,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));

  request.token_ids = tokens.data();
  request.n_tokens = emel::batch::planner::action::MAX_PLAN_STEPS + 1;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));

  request.n_tokens = 1;
  request.mode = emel::batch::planner::event::plan_mode::simple;
  CHECK(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                       planner_ctx));
}

TEST_CASE("batch_planner_guard_inputs_reject_invalid_metadata") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int8_t, 2> output_mask = {{1, 0}};
  std::array<uint64_t, 2> masks = {{0U, 1U}};
  std::array<int32_t, 2> primary_ids = {{0, 128}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::simple,
    .seq_mask_words = 0,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  request.seq_mask_words = emel::batch::planner::action::SEQ_WORDS + 1;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  request.seq_mask_words = 1;

  request.output_mask = output_mask.data();
  request.output_mask_count = 1;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  request.output_mask_count = static_cast<int32_t>(output_mask.size());

  request.seq_masks = masks.data();
  request.seq_masks_count = 1;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  request.seq_masks_count = static_cast<int32_t>(masks.size());

  request.seq_primary_ids = primary_ids.data();
  request.seq_primary_ids_count = 1;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  request.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());

  primary_ids[1] = 64;
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  primary_ids[1] = 0;

  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));
  masks[0] = 1U;
  CHECK(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                       planner_ctx));
}

TEST_CASE("batch_planner_guard_equal_sequential_requires_primary_ids") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};
  std::array<int32_t, 2> primary_ids = {{0, 1}};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .mode = emel::batch::planner::event::plan_mode::equal,
    .seq_masks = masks.data(),
    .seq_masks_count = static_cast<int32_t>(masks.size()),
    .seq_primary_ids = nullptr,
    .equal_sequential = true,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));

  masks[1] = 3U;
  request.seq_primary_ids = primary_ids.data();
  request.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());
  CHECK_FALSE(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                             planner_ctx));

  masks[1] = 2U;
  CHECK(emel::batch::planner::guard::inputs_are_valid(make_runtime(request, request_ctx),
                                                       planner_ctx));
}

TEST_CASE("batch_planner_guard_mode_selection") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  CHECK(emel::batch::planner::guard::mode_is_simple(make_runtime(request, request_ctx),
                                                     planner_ctx));
  CHECK_FALSE(emel::batch::planner::guard::mode_is_equal(make_runtime(request, request_ctx),
                                                          planner_ctx));
  CHECK_FALSE(emel::batch::planner::guard::mode_is_seq(make_runtime(request, request_ctx),
                                                        planner_ctx));

  request.mode = emel::batch::planner::event::plan_mode::equal;
  CHECK(emel::batch::planner::guard::mode_is_equal(make_runtime(request, request_ctx),
                                                    planner_ctx));

  request.mode = emel::batch::planner::event::plan_mode::seq;
  CHECK(emel::batch::planner::guard::mode_is_seq(make_runtime(request, request_ctx),
                                                  planner_ctx));
}

TEST_CASE("batch_planner_guard_planning_failed") {
  emel::batch::planner::action::context planner_ctx{};
  emel::batch::planner::event::request_ctx request_ctx{};
  done_capture done{};
  error_capture error{};

  emel::batch::planner::event::request request{
    .n_tokens = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  request_ctx.step_count = 0;
  request_ctx.total_outputs = 1;
  CHECK(emel::batch::planner::guard::planning_failed(make_runtime(request, request_ctx),
                                                      planner_ctx));

  request_ctx.step_count = 1;
  request_ctx.total_outputs = 1;
  request_ctx.token_indices_count = 1;
  request_ctx.step_token_offsets[1] = 1;
  CHECK_FALSE(emel::batch::planner::guard::planning_failed(make_runtime(request, request_ctx),
                                                            planner_ctx));

  request.n_tokens = 2;
  request_ctx.token_indices_count = 1;
  CHECK(emel::batch::planner::guard::planning_failed(make_runtime(request, request_ctx),
                                                      planner_ctx));

  request_ctx.token_indices_count = request.n_tokens;
  request_ctx.step_count = emel::batch::planner::action::MAX_PLAN_STEPS + 1;
  CHECK(emel::batch::planner::guard::planning_failed(make_runtime(request, request_ctx),
                                                      planner_ctx));
}
