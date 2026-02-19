#include <array>
#include <cstdint>
#include <vector>
#include <doctest/doctest.h>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct done_capture {
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t calls = 0;

  void on_done(const emel::batch::splitter::events::splitting_done & ev) noexcept {
    calls += 1;
    ubatch_count = ev.ubatch_count;
    total_outputs = ev.total_outputs;
  }
};

struct error_capture {
  int32_t err = EMEL_OK;
  int32_t calls = 0;

  void on_error(const emel::batch::splitter::events::splitting_error & ev) noexcept {
    calls += 1;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::splitter::events::splitting_done &)> make_done(
    done_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
    done_capture,
    &done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::splitter::events::splitting_error &)> make_error(
    error_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
    error_capture,
    &error_capture::on_error>(capture);
}

}  // namespace

TEST_CASE("batch_splitter_actions_begin_split_copies_request") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<uint64_t, 3> masks = {{1U, 2U, 3U}};
  std::array<int32_t, 3> primary_ids = {{0, 1, 2}};
  std::array<int8_t, 3> outputs = {{1, 0, 1}};

  emel::batch::splitter::event::split request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .seq_masks = masks.data(),
    .seq_masks_count = static_cast<int32_t>(masks.size()),
    .seq_primary_ids = primary_ids.data(),
    .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
    .equal_sequential = false,
    .seq_mask_words = 1,
    .output_mask = outputs.data(),
    .output_mask_count = static_cast<int32_t>(outputs.size()),
    .output_all = true,
  };

  emel::batch::splitter::action::begin_split(request, ctx);

  CHECK(ctx.token_ids == tokens.data());
  CHECK(ctx.n_tokens == 3);
  CHECK(ctx.requested_n_ubatch == 2);
  CHECK(ctx.mode == emel::batch::splitter::event::split_mode::seq);
  CHECK(ctx.seq_masks == masks.data());
  CHECK(ctx.seq_masks_count == static_cast<int32_t>(masks.size()));
  CHECK(ctx.seq_primary_ids == primary_ids.data());
  CHECK(ctx.seq_primary_ids_count == static_cast<int32_t>(primary_ids.size()));
  CHECK(ctx.equal_sequential == false);
  CHECK(ctx.seq_mask_words == 1);
  CHECK(ctx.output_mask == outputs.data());
  CHECK(ctx.output_mask_count == static_cast<int32_t>(outputs.size()));
  CHECK(ctx.output_all);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_normalize_batch_clamps_requested") {
  emel::batch::splitter::action::context ctx{};

  ctx.n_tokens = 4;
  ctx.requested_n_ubatch = 0;
  emel::batch::splitter::action::normalize_batch(ctx);
  CHECK(ctx.effective_n_ubatch == 4);

  ctx.requested_n_ubatch = 10;
  emel::batch::splitter::action::normalize_batch(ctx);
  CHECK(ctx.effective_n_ubatch == 4);
}

TEST_CASE("batch_splitter_actions_sequence_mask_normalization_variants") {
  emel::batch::splitter::action::context ctx{};
  std::array<uint64_t, 3> seq_masks = {{7U, 1U, 2U}};
  std::array<int32_t, 3> seq_primary_ids = {{2, 1, 5}};

  ctx.seq_masks = seq_masks.data();
  ctx.seq_primary_ids = nullptr;

  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 0)[0] == 7U);
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 1)[0] == 1U);
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 2)[0] == 2U);

  ctx.seq_masks = nullptr;
  ctx.seq_primary_ids = seq_primary_ids.data();

  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 0)[0] == (uint64_t{1} << 2));
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 1)[0] == (uint64_t{1} << 1));
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 2)[0] == (uint64_t{1} << 5));
}

TEST_CASE("batch_splitter_actions_push_ubatch_size_limits") {
  emel::batch::splitter::action::context ctx{};

  CHECK_FALSE(emel::batch::splitter::action::push_ubatch_size(ctx, 0));

  ctx.ubatch_count = emel::batch::splitter::action::MAX_UBATCHES;
  CHECK_FALSE(emel::batch::splitter::action::push_ubatch_size(ctx, 1));
}

TEST_CASE("batch_splitter_actions_create_ubatches_simple_success") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = 2;

  emel::batch::splitter::action::create_ubatches_simple(ctx);
  CHECK(ctx.ubatch_count == 2);
  CHECK(ctx.ubatch_sizes[0] == 2);
  CHECK(ctx.ubatch_sizes[1] == 2);
}

TEST_CASE("batch_splitter_actions_create_ubatches_equal_without_masks") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.effective_n_ubatch = 2;
  ctx.seq_masks = nullptr;
  ctx.seq_primary_ids = nullptr;

  emel::batch::splitter::action::create_ubatches_equal(ctx);
  CHECK(ctx.ubatch_count == 3);
  CHECK(ctx.ubatch_sizes[0] == 2);
  CHECK(ctx.ubatch_sizes[1] == 2);
  CHECK(ctx.ubatch_sizes[2] == 1);
}

TEST_CASE("batch_splitter_actions_create_ubatches_equal_skips_nonconsecutive_primary") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<uint64_t, 3> masks = {{1U, 2U, 4U}};
  std::array<int32_t, 3> primary_ids = {{0, 2, 1}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.effective_n_ubatch = 2;
  ctx.seq_masks = masks.data();
  ctx.seq_primary_ids = primary_ids.data();
  ctx.equal_sequential = true;

  emel::batch::splitter::action::create_ubatches_equal(ctx);
  CHECK(ctx.ubatch_count == 2);
  CHECK(ctx.ubatch_sizes[0] == 2);
  CHECK(ctx.ubatch_sizes[1] == 1);
}

TEST_CASE("batch_splitter_actions_create_ubatches_seq_with_masks") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{3U, 1U, 2U, 1U}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  ctx.effective_n_ubatch = 3;
  ctx.seq_masks = masks.data();

  emel::batch::splitter::action::create_ubatches_seq(ctx);
  CHECK(ctx.ubatch_count == 2);
  CHECK(ctx.ubatch_sizes[0] == 3);
  CHECK(ctx.ubatch_sizes[1] == 1);
}

TEST_CASE("batch_splitter_actions_mask_helpers_cover_edges") {
  using emel::batch::splitter::action::mask_any_set;
  using emel::batch::splitter::action::mask_equal;
  using emel::batch::splitter::action::mask_has_multiple_bits;
  using emel::batch::splitter::action::mask_is_subset;
  using emel::batch::splitter::action::mask_overlaps;
  using emel::batch::splitter::action::seq_mask_t;

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

  if constexpr (emel::batch::splitter::action::SEQ_WORDS > 1) {
    seq_mask_t spread = {};
    spread[0] = 1U;
    spread[1] = 1U;
    CHECK(mask_has_multiple_bits(spread));
  }
}

TEST_CASE("batch_splitter_actions_count_total_outputs_variants") {
  emel::batch::splitter::action::context ctx{};
  std::array<int8_t, 3> output_mask = {{1, 0, 1}};

  ctx.n_tokens = 3;
  ctx.output_all = true;
  CHECK(emel::batch::splitter::action::count_total_outputs(ctx) == 3);

  ctx.output_all = false;
  ctx.output_mask = nullptr;
  CHECK(emel::batch::splitter::action::count_total_outputs(ctx) == 1);

  ctx.output_mask = output_mask.data();
  ctx.output_mask_count = static_cast<int32_t>(output_mask.size());
  CHECK(emel::batch::splitter::action::count_total_outputs(ctx) == 2);
}

TEST_CASE("batch_splitter_actions_reject_overflow_helpers") {
  emel::batch::splitter::action::context ctx{};

  ctx.token_indices_count = emel::batch::splitter::action::MAX_UBATCHES;
  CHECK_FALSE(emel::batch::splitter::action::append_token_index(ctx, 0));

  ctx.ubatch_count = emel::batch::splitter::action::MAX_UBATCHES;
  CHECK_FALSE(emel::batch::splitter::action::begin_ubatch(ctx));
}

TEST_CASE("batch_splitter_actions_create_ubatches_simple_fails_on_index_overflow") {
  emel::batch::splitter::action::context ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::splitter::action::MAX_UBATCHES) + 1U;
  std::vector<int32_t> tokens(token_count, 1);

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = static_cast<int32_t>(tokens.size());

  emel::batch::splitter::action::create_ubatches_simple(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_create_ubatches_simple_fails_on_ubatch_overflow") {
  emel::batch::splitter::action::context ctx{};
  const size_t token_count =
      static_cast<size_t>(emel::batch::splitter::action::MAX_UBATCHES) + 1U;
  std::vector<int32_t> tokens(token_count, 1);

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = 1;

  emel::batch::splitter::action::create_ubatches_simple(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_create_ubatches_equal_rejects_zero_batch") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.effective_n_ubatch = 0;

  emel::batch::splitter::action::create_ubatches_equal(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_create_ubatches_equal_fails_when_groups_exceed_capacity") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.effective_n_ubatch = 1;
  ctx.seq_masks = masks.data();
  ctx.seq_masks_count = static_cast<int32_t>(masks.size());
  ctx.seq_mask_words = 1;

  emel::batch::splitter::action::create_ubatches_equal(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_create_ubatches_seq_without_masks_failure") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  ctx.effective_n_ubatch = 0;
  ctx.seq_masks = nullptr;

  emel::batch::splitter::action::create_ubatches_seq(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_create_ubatches_failures_reset_outputs") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = 0;

  emel::batch::splitter::action::create_ubatches_simple(ctx);
  CHECK(ctx.ubatch_count == 0);
  CHECK(ctx.total_outputs == 0);
}

TEST_CASE("batch_splitter_actions_dispatch_helpers_cover_callbacks") {
  emel::batch::splitter::action::context ctx{};
  done_capture done{};
  error_capture error{};

  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_count = 1;
  ctx.total_outputs = 2;

  emel::batch::splitter::event::split request{
    .on_done = make_done(&done),
    .on_error = make_error(&error),
  };

  emel::batch::splitter::action::dispatch_done(request, ctx);
  CHECK(done.calls == 1);
  CHECK(done.ubatch_count == 1);
  CHECK(done.total_outputs == 2);

  emel::batch::splitter::action::dispatch_invalid_request(request);
  CHECK(error.calls == 1);
  CHECK(error.err == EMEL_ERR_INVALID_ARGUMENT);

  emel::batch::splitter::action::dispatch_split_failed(request);
  CHECK(error.calls == 2);
  CHECK(error.err == EMEL_ERR_BACKEND);

  emel::batch::splitter::action::dispatch_unexpected(request);
  CHECK(error.calls == 3);
  CHECK(error.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_actions_dispatch_helpers_skip_missing_callbacks") {
  emel::batch::splitter::action::context ctx{};
  emel::batch::splitter::event::split request{};

  CHECK_FALSE(static_cast<bool>(request.on_done));
  CHECK_FALSE(static_cast<bool>(request.on_error));

  emel::batch::splitter::action::dispatch_done(request, ctx);
  emel::batch::splitter::action::dispatch_invalid_request(request);
  emel::batch::splitter::action::dispatch_split_failed(request);
  emel::batch::splitter::action::dispatch_unexpected(request);
  CHECK(true);
}
