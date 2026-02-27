#include <array>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/detail.hpp"

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

TEST_CASE("batch_planner_modes_detail_sequence_mask_normalization_variants") {
  done_capture done{};
  error_capture error{};
  std::array<uint64_t, 3> seq_masks = {{7U, 1U, 2U}};
  std::array<int32_t, 3> seq_primary_ids = {{2, 1, 5}};

  emel::batch::planner::event::request request{
    .n_tokens = 3,
    .seq_masks = seq_masks.data(),
    .seq_primary_ids = nullptr,
    .seq_mask_words = 1,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 0)[0] == 7U);
  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 1)[0] == 1U);
  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 2)[0] == 2U);

  request.seq_masks = nullptr;
  request.seq_primary_ids = seq_primary_ids.data();

  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 0)[0] ==
        (uint64_t{1} << 2));
  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 1)[0] ==
        (uint64_t{1} << 1));
  CHECK(emel::batch::planner::modes::detail::normalized_seq_mask(request, 2)[0] ==
        (uint64_t{1} << 5));
}

TEST_CASE("batch_planner_modes_detail_push_step_size_limits") {
  emel::batch::planner::event::request_ctx ctx{};

  CHECK_FALSE(emel::batch::planner::modes::detail::push_step_size(ctx, 0));

  ctx.step_count = emel::batch::planner::action::MAX_PLAN_STEPS;
  CHECK_FALSE(emel::batch::planner::modes::detail::push_step_size(ctx, 1));
}

TEST_CASE("batch_planner_modes_detail_mask_helpers_cover_edges") {
  using emel::batch::planner::modes::detail::mask_any_set;
  using emel::batch::planner::modes::detail::mask_equal;
  using emel::batch::planner::modes::detail::mask_has_multiple_bits;
  using emel::batch::planner::modes::detail::mask_is_subset;
  using emel::batch::planner::modes::detail::mask_overlaps;
  using emel::batch::planner::modes::detail::seq_mask_t;

  seq_mask_t none = {};
  CHECK_FALSE(mask_any_set(none));

  seq_mask_t single = {};
  single[0] = 1U;
  CHECK(mask_any_set(single));

  seq_mask_t other = {};
  other[0] = 2U;
  CHECK_FALSE(mask_overlaps(single, other));

  seq_mask_t multi = {};
  multi[0] = 3U;
  CHECK(mask_overlaps(single, multi));
  CHECK_FALSE(mask_equal(single, multi));
  CHECK(mask_is_subset(multi, single));
  CHECK_FALSE(mask_is_subset(single, multi));
  CHECK(mask_has_multiple_bits(multi));

  if constexpr (emel::batch::planner::action::SEQ_WORDS > 1) {
    seq_mask_t spread = {};
    spread[0] = 1U;
    spread[1] = 1U;
    CHECK(mask_has_multiple_bits(spread));
  }
}

TEST_CASE("batch_planner_modes_detail_count_total_outputs_variants") {
  done_capture done{};
  error_capture error{};
  std::array<int8_t, 3> output_mask = {{1, 0, 1}};

  emel::batch::planner::event::request request{
    .n_tokens = 3,
    .output_all = true,
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };
  CHECK(emel::batch::planner::modes::detail::count_total_outputs(request) == 3);

  request.output_all = false;
  request.output_mask = nullptr;
  CHECK(emel::batch::planner::modes::detail::count_total_outputs(request) == 1);

  request.output_mask = output_mask.data();
  request.output_mask_count = static_cast<int32_t>(output_mask.size());
  CHECK(emel::batch::planner::modes::detail::count_total_outputs(request) == 2);
}

TEST_CASE("batch_planner_modes_detail_reject_overflow_helpers") {
  emel::batch::planner::event::request_ctx ctx{};

  ctx.token_indices_count = emel::batch::planner::action::MAX_PLAN_STEPS;
  CHECK_FALSE(emel::batch::planner::modes::detail::append_token_index(ctx, 0));

  ctx.step_count = emel::batch::planner::action::MAX_PLAN_STEPS;
  CHECK_FALSE(emel::batch::planner::modes::detail::begin_step(ctx));
}
