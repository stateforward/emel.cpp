#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"

TEST_CASE("batch_splitter_rejects_invalid_token_counts") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> ubatch_sizes = {{0, 0, 0, 0}};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("batch_splitter_rejects_invalid_capacity") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<int32_t, 1> ubatch_sizes = {{0}};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("batch_splitter_equal_mode_with_seq_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  std::array<int32_t, 8> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .seq_masks = masks.data(),
    .seq_primary_ids = primary_ids.data(),
    .equal_sequential = true,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
  }));

  CHECK(ubatch_count > 0);
  CHECK(total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_splitter_seq_mode_with_seq_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  std::array<int32_t, 8> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .seq_masks = masks.data(),
    .seq_primary_ids = primary_ids.data(),
    .equal_sequential = true,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
  }));

  CHECK(ubatch_count > 0);
  CHECK(total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_splitter_rejects_unknown_mode") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 4> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = static_cast<emel::batch::splitter::event::split_mode>(99),
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("batch_splitter_create_ubatches_rejects_invalid_mode") {
  emel::batch::splitter::action::context ctx{};
  ctx.mode = static_cast<emel::batch::splitter::event::split_mode>(99);
  ctx.n_tokens = 4;
  ctx.effective_n_ubatch = 2;
  int32_t err = EMEL_OK;

  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_create_ubatches_reports_simple_push_failure") {
  emel::batch::splitter::action::context ctx{};
  ctx.mode = emel::batch::splitter::event::split_mode::simple;
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  ctx.effective_n_ubatch = 1;
  int32_t err = EMEL_OK;

  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);

  CHECK(err != EMEL_OK);
}

TEST_CASE("batch_splitter_create_ubatches_equal_mode_rejects_large_chunk_count") {
  emel::batch::splitter::action::context ctx{};
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  ctx.effective_n_ubatch = 1;
  int32_t err = EMEL_OK;

  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("batch_splitter_create_ubatches_seq_mode_reports_push_failure") {
  emel::batch::splitter::action::context ctx{};
  ctx.mode = emel::batch::splitter::event::split_mode::seq;
  ctx.n_tokens = emel::batch::splitter::action::MAX_UBATCHES + 1;
  ctx.effective_n_ubatch = 1;
  int32_t err = EMEL_OK;

  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);

  CHECK(err != EMEL_OK);
}

TEST_CASE("batch_splitter_create_ubatches_equal_mode_seq_masks_overflows") {
  emel::batch::splitter::action::context ctx{};
  std::array<uint64_t, emel::batch::splitter::action::MAX_UBATCHES + 1> masks = {};
  ctx.mode = emel::batch::splitter::event::split_mode::equal;
  ctx.n_tokens = static_cast<int32_t>(masks.size());
  ctx.effective_n_ubatch = 1;
  ctx.seq_masks = masks.data();
  int32_t err = EMEL_OK;

  emel::batch::splitter::action::run_create_ubatches(
    emel::batch::splitter::event::create_ubatches{.error_out = &err},
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("batch_splitter_split_rejects_negative_capacity") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = -1,
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("batch_splitter_split_rejects_missing_sizes_buffer") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 1,
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}
