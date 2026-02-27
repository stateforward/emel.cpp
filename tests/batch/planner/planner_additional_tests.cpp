#include <algorithm>
#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct plan_capture {
  std::array<int32_t, 8> sizes = {};
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::planner::events::plan_done & ev) noexcept {
    done_called = true;
    err = EMEL_OK;
    step_count = ev.step_count;
    total_outputs = ev.total_outputs;
    if (ev.step_sizes == nullptr) {
      return;
    }
    const int32_t count = std::min<int32_t>(step_count, static_cast<int32_t>(sizes.size()));
    for (int32_t i = 0; i < count; ++i) {
      sizes[i] = ev.step_sizes[i];
    }
  }

  void on_error(const emel::batch::planner::events::plan_error & ev) noexcept {
    error_called = true;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::planner::events::plan_done &)> make_done(
    plan_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
      plan_capture,
      &plan_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::events::plan_error &)> make_error(
    plan_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
      plan_capture,
      &plan_capture::on_error>(capture);
}

}  // namespace

TEST_CASE("batch_planner_rejects_invalid_token_counts") {
  emel::batch::planner::sm machine{};
  plan_capture capture{};

  CHECK_FALSE(machine.process_event(emel::batch::planner::event::request{
      .token_ids = nullptr,
      .n_tokens = 0,
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::simple,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_token_data));
}

TEST_CASE("batch_planner_equal_mode_with_seq_masks") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::equal,
      .seq_masks = masks.data(),
      .seq_masks_count = static_cast<int32_t>(masks.size()),
      .seq_primary_ids = primary_ids.data(),
      .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
      .equal_sequential = true,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_planner_seq_mode_with_seq_masks") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> masks = {{1, 1, 2, 2}};
  std::array<int32_t, 4> primary_ids = {{0, 0, 1, 1}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::seq,
      .seq_masks = masks.data(),
      .seq_masks_count = static_cast<int32_t>(masks.size()),
      .seq_primary_ids = primary_ids.data(),
      .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
      .equal_sequential = true,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.total_outputs == static_cast<int32_t>(tokens.size()));
}

TEST_CASE("batch_planner_equal_mode_rejects_coupled_sequences_when_sequential") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 2> masks = {{3U, 1U}};
  std::array<int32_t, 2> primary_ids = {{0, 1}};
  plan_capture capture{};

  CHECK_FALSE(machine.process_event(emel::batch::planner::event::request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::equal,
      .seq_masks = masks.data(),
      .seq_masks_count = static_cast<int32_t>(masks.size()),
      .seq_primary_ids = primary_ids.data(),
      .seq_primary_ids_count = static_cast<int32_t>(primary_ids.size()),
      .equal_sequential = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::multiple_bits_in_mask));
}

TEST_CASE("batch_planner_supports_multiword_sequence_masks") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<uint64_t, 4> masks = {{0U, 1U, 0U, 2U}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::equal,
      .seq_masks = masks.data(),
      .seq_masks_count = static_cast<int32_t>(tokens.size()),
      .equal_sequential = false,
      .seq_mask_words = 2,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));
  CHECK(capture.done_called);
  CHECK(capture.step_count == 1);
  CHECK(capture.sizes[0] == 2);
}

TEST_CASE("batch_planner_rejects_unknown_mode") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  plan_capture capture{};

  CHECK_FALSE(machine.process_event(emel::batch::planner::event::request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = static_cast<emel::batch::planner::event::plan_mode>(99),
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_mode));
}
