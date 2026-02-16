#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/batch/splitter/sm.hpp"

namespace {

TEST_CASE("batch_splitter_sm_successful_split") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 2> ubatch_sizes = {{0, 0}};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = static_cast<int32_t>(ubatch_sizes.size()),
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
  CHECK(ubatch_count == 2);
}

TEST_CASE("batch_splitter_sm_reports_capacity_error") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 1> ubatch_sizes = {{0}};
  int32_t err = EMEL_OK;

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes.data(),
    .ubatch_sizes_capacity = 0,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

TEST_CASE("batch_splitter_sm_validation_error_path") {
  emel::batch::splitter::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
