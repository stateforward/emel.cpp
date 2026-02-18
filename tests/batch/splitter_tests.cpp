#include <algorithm>
#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/splitter/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct split_capture {
  std::array<int32_t, 16> sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::splitter::events::splitting_done & ev) noexcept {
    done_called = true;
    err = EMEL_OK;
    ubatch_count = ev.ubatch_count;
    total_outputs = ev.total_outputs;
    if (ev.ubatch_sizes == nullptr) {
      return;
    }
    const int32_t count = std::min<int32_t>(
      ubatch_count,
      static_cast<int32_t>(sizes.size()));
    for (int32_t i = 0; i < count; ++i) {
      sizes[i] = ev.ubatch_sizes[i];
    }
  }

  void on_error(const emel::batch::splitter::events::splitting_error & ev) noexcept {
    error_called = true;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::splitter::events::splitting_done &)> make_done(
    split_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
    split_capture,
    &split_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::splitter::events::splitting_error &)> make_error(
    split_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
    split_capture,
    &split_capture::on_error>(capture);
}

}  // namespace

TEST_CASE("batch_splitter_starts_initialized") {
  emel::batch::splitter::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::splitter::initialized>));
}

TEST_CASE("batch_splitter_splits_tokens_into_ubatches") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.err == EMEL_OK);
  CHECK(capture.ubatch_count == 3);
  CHECK(capture.total_outputs == 5);
  CHECK(capture.sizes[0] == 2);
  CHECK(capture.sizes[1] == 2);
  CHECK(capture.sizes[2] == 1);
}

TEST_CASE("batch_splitter_equal_mode_balances_chunk_sizes") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 10> tokens = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 4,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.ubatch_count == 3);
  CHECK(capture.sizes[0] == 4);
  CHECK(capture.sizes[1] == 3);
  CHECK(capture.sizes[2] == 3);
}

TEST_CASE("batch_splitter_seq_mode_uses_sequential_chunking") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 7> tokens = {{1, 2, 3, 4, 5, 6, 7}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 3,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.ubatch_count == 3);
  CHECK(capture.sizes[0] == 3);
  CHECK(capture.sizes[1] == 3);
  CHECK(capture.sizes[2] == 1);
}

TEST_CASE("batch_splitter_equal_mode_supports_sequence_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 6> tokens = {{1, 2, 3, 4, 5, 6}};
  std::array<uint64_t, 6> seq_masks = {{1U, 2U, 1U, 2U, 1U, 2U}};
  std::array<int32_t, 6> seq_primary_ids = {{0, 1, 0, 1, 0, 1}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 4,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .seq_masks = seq_masks.data(),
    .seq_primary_ids = seq_primary_ids.data(),
    .equal_sequential = true,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.ubatch_count == 2);
  CHECK(capture.sizes[0] == 4);
  CHECK(capture.sizes[1] == 2);
  CHECK(capture.total_outputs == 6);
}

TEST_CASE("batch_splitter_seq_mode_supports_sequence_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> seq_masks = {{3U, 1U, 2U, 1U}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 3,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .seq_masks = seq_masks.data(),
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.ubatch_count == 2);
  CHECK(capture.sizes[0] == 3);
  CHECK(capture.sizes[1] == 1);
  CHECK(capture.total_outputs == 4);
}

TEST_CASE("batch_splitter_reports_invalid_arguments") {
  emel::batch::splitter::sm machine{};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 4,
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  split_capture mode_capture{};
  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = static_cast<emel::batch::splitter::event::split_mode>(99),
    .on_done = make_done(&mode_capture),
    .on_error = make_error(&mode_capture),
  }));
  CHECK(mode_capture.error_called);
  CHECK(mode_capture.err == EMEL_ERR_INVALID_ARGUMENT);
}
