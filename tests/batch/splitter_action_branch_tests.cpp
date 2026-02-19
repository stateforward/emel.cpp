#include <array>
#include <doctest/doctest.h>

#include "emel/batch/splitter/guards.hpp"
#include "emel/callback.hpp"

namespace {

inline void noop_done(const emel::batch::splitter::events::splitting_done &) noexcept {}
inline void noop_error(const emel::batch::splitter::events::splitting_error &) noexcept {}

}  // namespace

TEST_CASE("batch_splitter_guard_callbacks_are_valid") {
  emel::batch::splitter::event::split request{};
  CHECK_FALSE(emel::batch::splitter::guard::callbacks_are_valid(request));

  request.on_done =
    emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
      &noop_done>();
  request.on_error =
    emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
      &noop_error>();
  CHECK(emel::batch::splitter::guard::callbacks_are_valid(request));
}

TEST_CASE("batch_splitter_guard_inputs_are_valid") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 1> tokens = {{1}};

  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.token_ids = tokens.data();
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.n_tokens = 1;
  ctx.mode = static_cast<emel::batch::splitter::event::split_mode>(99);
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK(emel::batch::splitter::guard::inputs_are_valid(ctx));
}

TEST_CASE("batch_splitter_guard_inputs_reject_invalid_metadata") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int8_t, 2> output_mask = {{1, 0}};
  std::array<uint64_t, 2> masks = {{0U, 1U}};
  std::array<int32_t, 2> primary_ids = {{0, 128}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.seq_mask_words = 0;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  ctx.seq_mask_words = emel::batch::splitter::action::SEQ_WORDS + 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  ctx.seq_mask_words = 1;

  ctx.output_mask = output_mask.data();
  ctx.output_mask_count = 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  ctx.output_mask_count = static_cast<int32_t>(output_mask.size());

  ctx.seq_masks = masks.data();
  ctx.seq_masks_count = 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  ctx.seq_masks_count = static_cast<int32_t>(masks.size());

  ctx.seq_primary_ids = primary_ids.data();
  ctx.seq_primary_ids_count = 1;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  ctx.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());

  primary_ids[1] = 64;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  primary_ids[1] = 0;

  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));
  masks[0] = 1U;
  CHECK(emel::batch::splitter::guard::inputs_are_valid(ctx));
}

TEST_CASE("batch_splitter_guard_equal_sequential_requires_primary_ids") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};
  std::array<int32_t, 2> primary_ids = {{0, 1}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.equal_sequential = true;
  ctx.seq_mask_words = 1;

  ctx.seq_masks = masks.data();
  ctx.seq_masks_count = static_cast<int32_t>(masks.size());
  ctx.seq_primary_ids = nullptr;
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  masks[1] = 3U;
  ctx.seq_primary_ids = primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());
  CHECK_FALSE(emel::batch::splitter::guard::inputs_are_valid(ctx));

  masks[1] = 2U;
  CHECK(emel::batch::splitter::guard::inputs_are_valid(ctx));
}

TEST_CASE("batch_splitter_guard_mode_selection") {
  emel::batch::splitter::action::context ctx{};

  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  CHECK(emel::batch::splitter::guard::mode_is_simple(ctx));
  CHECK_FALSE(emel::batch::splitter::guard::mode_is_equal(ctx));
  CHECK_FALSE(emel::batch::splitter::guard::mode_is_seq(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  CHECK(emel::batch::splitter::guard::mode_is_equal(ctx));

  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  CHECK(emel::batch::splitter::guard::mode_is_seq(ctx));
}

TEST_CASE("batch_splitter_guard_split_failed") {
  emel::batch::splitter::action::context ctx{};

  ctx.ubatch_count = 0;
  ctx.total_outputs = 1;
  CHECK(emel::batch::splitter::guard::split_failed(ctx));

  ctx.n_tokens = 1;
  ctx.ubatch_count = 1;
  ctx.total_outputs = 1;
  ctx.token_indices_count = 1;
  ctx.ubatch_token_offsets[1] = 1;
  CHECK_FALSE(emel::batch::splitter::guard::split_failed(ctx));

  ctx.n_tokens = 2;
  ctx.token_indices_count = 1;
  CHECK(emel::batch::splitter::guard::split_failed(ctx));

  ctx.token_indices_count = ctx.n_tokens;
  ctx.ubatch_count = emel::batch::splitter::action::MAX_UBATCHES + 1;
  CHECK(emel::batch::splitter::guard::split_failed(ctx));
}
