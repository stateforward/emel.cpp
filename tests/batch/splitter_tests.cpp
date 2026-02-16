#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"

TEST_CASE("batch_splitter_starts_initialized") {
  emel::batch::splitter::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::splitter::initialized>));
}

TEST_CASE("batch_splitter_splits_tokens_into_ubatches") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};
  std::array<int32_t, 8> ubatch_sizes = {{0, 0, 0, 0, 0, 0, 0, 0}};
  int32_t ubatch_count = 0;
  int32_t outputs_total = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &outputs_total,
  }));

  CHECK(ubatch_count == 3);
  CHECK(outputs_total == 5);
  CHECK(ubatch_sizes[0] == 2);
  CHECK(ubatch_sizes[1] == 2);
  CHECK(ubatch_sizes[2] == 1);
}

TEST_CASE("batch_splitter_equal_mode_balances_chunk_sizes") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 10> tokens = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  std::array<int32_t, 16> ubatch_sizes = {};
  int32_t ubatch_count = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 4,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
  }));

  CHECK(ubatch_count == 3);
  CHECK(ubatch_sizes[0] == 4);
  CHECK(ubatch_sizes[1] == 3);
  CHECK(ubatch_sizes[2] == 3);
}

TEST_CASE("batch_splitter_seq_mode_uses_sequential_chunking") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 7> tokens = {{1, 2, 3, 4, 5, 6, 7}};
  std::array<int32_t, 8> ubatch_sizes = {};
  int32_t ubatch_count = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 3,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
  }));

  CHECK(ubatch_count == 3);
  CHECK(ubatch_sizes[0] == 3);
  CHECK(ubatch_sizes[1] == 3);
  CHECK(ubatch_sizes[2] == 1);
}

TEST_CASE("batch_splitter_equal_mode_supports_sequence_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 6> tokens = {{1, 2, 3, 4, 5, 6}};
  std::array<uint64_t, 6> seq_masks = {{1U, 2U, 1U, 2U, 1U, 2U}};
  std::array<int32_t, 6> seq_primary_ids = {{0, 1, 0, 1, 0, 1}};
  std::array<int32_t, 8> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t outputs_total = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 4,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .seq_masks = seq_masks.data(),
    .seq_primary_ids = seq_primary_ids.data(),
    .equal_sequential = true,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &outputs_total,
  }));

  CHECK(ubatch_count == 2);
  CHECK(ubatch_sizes[0] == 4);
  CHECK(ubatch_sizes[1] == 2);
  CHECK(outputs_total == 6);
}

TEST_CASE("batch_splitter_seq_mode_supports_sequence_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> seq_masks = {{3U, 1U, 2U, 1U}};
  std::array<int32_t, 8> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t outputs_total = 0;

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 3,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .seq_masks = seq_masks.data(),
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &outputs_total,
  }));

  CHECK(ubatch_count == 2);
  CHECK(ubatch_sizes[0] == 3);
  CHECK(ubatch_sizes[1] == 1);
  CHECK(outputs_total == 4);
}

TEST_CASE("batch_splitter_reports_invalid_arguments") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> ubatch_sizes = {{0, 0, 0, 0}};
  int32_t ubatch_count = 0;
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 4,
    .n_ubatch = 2,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);

  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  error_out = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = static_cast<emel::batch::splitter::event::split_mode>(99),
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}

TEST_CASE("batch_splitter_reports_publish_capacity_checks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 3> tokens = {{7, 8, 9}};
  std::array<int32_t, 1> ubatch_sizes_small = {{0}};
  int32_t error_out = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .ubatch_sizes_out = ubatch_sizes_small.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes_small.size()),
    .error_out = &error_out,
  }));
  CHECK(error_out != EMEL_OK);
}
