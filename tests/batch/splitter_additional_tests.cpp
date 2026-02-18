#include <algorithm>
#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct split_capture {
  std::array<int32_t, 8> sizes = {};
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

TEST_CASE("batch_splitter_rejects_invalid_token_counts") {
  emel::batch::splitter::sm machine{};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_equal_mode_with_seq_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::equal,
    .seq_masks = masks.data(),
    .seq_primary_ids = primary_ids.data(),
    .equal_sequential = true,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_splitter_seq_mode_with_seq_masks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::seq,
    .seq_masks = masks.data(),
    .seq_primary_ids = primary_ids.data(),
    .equal_sequential = true,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_splitter_rejects_unknown_mode") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = static_cast<emel::batch::splitter::event::split_mode>(99),
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_split_rejects_missing_callbacks") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  split_capture capture{};

  CHECK(machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = {},
    .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);
}
