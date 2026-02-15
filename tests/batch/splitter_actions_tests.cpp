#include <array>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/emel.h"

TEST_CASE("batch_splitter_actions_handle_null_error_pointers") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.requested_n_ubatch = 1;
  ctx.effective_n_ubatch = 1;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;

  emel::batch::splitter::action::run_validate(
    emel::batch::splitter::event::validate{.error_out = nullptr},
    ctx);
  emel::batch::splitter::action::run_normalize_batch(
    emel::batch::splitter::event::normalize_batch{.error_out = nullptr},
    ctx);
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = nullptr},
    ctx);
  emel::batch::splitter::action::run_publish(
    emel::batch::splitter::event::publish{.error_out = nullptr},
    ctx);
}

TEST_CASE("batch_splitter_actions_validate_and_normalize_boundaries") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.ubatch_sizes_capacity = -1;
  emel::batch::splitter::action::run_validate(
    emel::batch::splitter::event::validate{.error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.requested_n_ubatch = 0;
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  err = EMEL_OK;
  emel::batch::splitter::action::run_normalize_batch(
    emel::batch::splitter::event::normalize_batch{.error_out = &err},
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.effective_n_ubatch == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_splitter_actions_sequence_mask_normalization_variants") {
  emel::batch::splitter::action::context ctx{};
  std::array<uint64_t, 3> seq_masks = {{7U, 0U, 0U}};
  std::array<int32_t, 3> seq_primary_ids = {{2, -1, 5}};

  ctx.seq_masks = seq_masks.data();
  ctx.seq_primary_ids = seq_primary_ids.data();

  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 0) == 7U);
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 1) == (uint64_t{1} << 1));
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 2) == (uint64_t{1} << 5));
}

TEST_CASE("batch_splitter_actions_push_ubatch_size_error_paths") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(emel::batch::splitter::action::push_ubatch_size(ctx, 0, &err));
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.ubatch_count = emel::batch::splitter::action::MAX_UBATCHES;
  err = EMEL_OK;
  CHECK_FALSE(emel::batch::splitter::action::push_ubatch_size(ctx, 1, &err));
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("batch_splitter_actions_create_ubatches_additional_paths") {
  emel::batch::splitter::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> seq_masks = {{1U, 2U, 4U, 8U}};
  std::array<int32_t, 4> seq_primary_ids = {{0, 2, 1, 3}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = 0;
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.seq_masks = seq_masks.data();
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.equal_sequential = true;
  ctx.effective_n_ubatch = 4;
  err = EMEL_OK;
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.ubatch_count == 2);
  CHECK(ctx.ubatch_sizes[0] == 2);
  CHECK(ctx.ubatch_sizes[1] == 2);

  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  ctx.seq_primary_ids = nullptr;
  ctx.effective_n_ubatch = 3;
  err = EMEL_OK;
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.ubatch_count >= 1);
}

TEST_CASE("batch_splitter_actions_publish_paths_without_buffer_copy") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  std::array<int32_t, 1> sizes_small = {{0}};

  ctx.ubatch_count = 2;
  ctx.ubatch_sizes[0] = 3;
  ctx.ubatch_sizes[1] = 1;
  ctx.total_outputs = 4;
  ctx.ubatch_sizes_out = nullptr;
  ctx.ubatch_count_out = &ubatch_count;
  ctx.total_outputs_out = &total_outputs;

  emel::batch::splitter::action::run_publish(
    emel::batch::splitter::event::publish{.error_out = &err},
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(ubatch_count == 2);
  CHECK(total_outputs == 4);

  ctx.ubatch_sizes_out = sizes_small.data();
  ctx.ubatch_sizes_capacity = static_cast<int32_t>(sizes_small.size());
  err = EMEL_OK;
  emel::batch::splitter::action::run_publish(
    emel::batch::splitter::event::publish{.error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
