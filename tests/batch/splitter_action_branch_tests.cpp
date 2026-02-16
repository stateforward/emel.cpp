#include <array>
#include <doctest/doctest.h>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/emel.h"

TEST_CASE("batch_splitter_run_validate_handles_invalid_inputs") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  emel::batch::splitter::action::run_validate(
    emel::batch::splitter::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> tokens = {{1}};
  ctx.token_ids = tokens.data();
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  err = EMEL_OK;
  emel::batch::splitter::action::run_validate(
    emel::batch::splitter::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.n_tokens = 1;
  ctx.mode = static_cast<emel::batch::splitter::event::split_mode>(99);
  err = EMEL_OK;
  emel::batch::splitter::action::run_validate(
    emel::batch::splitter::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_run_normalize_batch_clamps_requested") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.n_tokens = 4;
  ctx.requested_n_ubatch = 0;
  emel::batch::splitter::action::run_normalize_batch(
    emel::batch::splitter::event::normalize_batch{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.effective_n_ubatch == 4);

  ctx.requested_n_ubatch = 10;
  emel::batch::splitter::action::run_normalize_batch(
    emel::batch::splitter::event::normalize_batch{.error_out = &err}, ctx);
  CHECK(ctx.effective_n_ubatch == 4);
}

TEST_CASE("batch_splitter_run_create_ubatches_handles_simple_mode") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.n_tokens = 3;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.effective_n_ubatch = 2;
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.ubatch_count == 2);
}

TEST_CASE("batch_splitter_run_create_ubatches_rejects_invalid_effective") {
  emel::batch::splitter::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.n_tokens = 1;
  ctx.effective_n_ubatch = 0;
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_normalized_seq_mask_prefers_masks") {
  emel::batch::splitter::action::context ctx{};
  std::array<uint64_t, 1> masks = {{8}};

  ctx.seq_masks = masks.data();
  CHECK(emel::batch::splitter::action::normalized_seq_mask(ctx, 0) == 8);
}
