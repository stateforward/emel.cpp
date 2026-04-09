#include <algorithm>
#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/planner/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct plan_capture {
  std::array<int32_t, 16> sizes = {};
  int32_t step_count = 0;
  int32_t total_outputs = 0;
  int32_t err = emel::error::cast(emel::batch::planner::error::none);
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::planner::events::plan_done & ev) noexcept {
    done_called = true;
    err = emel::error::cast(emel::batch::planner::error::none);
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

TEST_CASE("batch_planner_starts_initialized") {
  emel::batch::planner::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::planner::state_idle>));
}

TEST_CASE("batch_planner_splits_tokens_into_steps") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 5> tokens = {{1, 2, 3, 4, 5}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::simple,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.err == emel::error::cast(emel::batch::planner::error::none));
  CHECK(capture.step_count == 3);
  CHECK(capture.total_outputs == 5);
  CHECK(capture.sizes[0] == 2);
  CHECK(capture.sizes[1] == 2);
  CHECK(capture.sizes[2] == 1);
}

TEST_CASE("batch_planner_equal_mode_fills_single_sequence_steps") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 10> tokens = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 4,
      .mode = emel::batch::planner::event::plan_mode::equal,
      .equal_sequential = false,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.step_count == 3);
  CHECK(capture.sizes[0] == 4);
  CHECK(capture.sizes[1] == 4);
  CHECK(capture.sizes[2] == 2);
}

TEST_CASE("batch_planner_seq_mode_uses_sequential_chunking") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 7> tokens = {{1, 2, 3, 4, 5, 6, 7}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 3,
      .mode = emel::batch::planner::event::plan_mode::seq,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.step_count == 3);
  CHECK(capture.sizes[0] == 3);
  CHECK(capture.sizes[1] == 3);
  CHECK(capture.sizes[2] == 1);
}

TEST_CASE("batch_planner_sequential_mode_aliases_seq") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 7> tokens = {{1, 2, 3, 4, 5, 6, 7}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 3,
      .mode = emel::batch::planner::event::plan_mode::sequential,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.step_count == 3);
  CHECK(capture.sizes[0] == 3);
  CHECK(capture.sizes[1] == 3);
  CHECK(capture.sizes[2] == 1);
}

TEST_CASE("batch_planner_equal_mode_supports_sequence_masks") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 6> tokens = {{1, 2, 3, 4, 5, 6}};
  std::array<uint64_t, 6> seq_masks = {{1U, 2U, 1U, 2U, 1U, 2U}};
  std::array<int32_t, 6> seq_primary_ids = {{0, 1, 0, 1, 0, 1}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 4,
      .mode = emel::batch::planner::event::plan_mode::equal,
      .seq_masks = seq_masks.data(),
      .seq_masks_count = static_cast<int32_t>(seq_masks.size()),
      .seq_primary_ids = seq_primary_ids.data(),
      .seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size()),
      .equal_sequential = true,
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.step_count == 2);
  CHECK(capture.sizes[0] == 4);
  CHECK(capture.sizes[1] == 2);
  CHECK(capture.total_outputs == 6);
}

TEST_CASE("batch_planner_seq_mode_supports_sequence_masks") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<uint64_t, 4> seq_masks = {{3U, 1U, 2U, 1U}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 3,
      .mode = emel::batch::planner::event::plan_mode::seq,
      .seq_masks = seq_masks.data(),
      .seq_masks_count = static_cast<int32_t>(seq_masks.size()),
      .output_all = true,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.step_count == 2);
  CHECK(capture.sizes[0] == 3);
  CHECK(capture.sizes[1] == 1);
  CHECK(capture.total_outputs == 4);
}

TEST_CASE("batch_planner_counts_outputs_with_output_mask") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<int8_t, 4> outputs = {{1, 0, 1, 0}};
  plan_capture capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::simple,
      .output_mask = outputs.data(),
      .output_mask_count = static_cast<int32_t>(outputs.size()),
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));

  CHECK(capture.done_called);
  CHECK(capture.total_outputs == 2);
}

TEST_CASE("batch_planner_reports_invalid_arguments") {
  emel::batch::planner::sm machine{};
  plan_capture capture{};

  CHECK_FALSE(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = nullptr,
      .n_tokens = 4,
      .n_steps = 2,
      .mode = emel::batch::planner::event::plan_mode::simple,
      .on_done = make_done(&capture),
      .on_error = make_error(&capture),
  }));
  CHECK(capture.error_called);
  CHECK(capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_token_data));

  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  plan_capture mode_capture{};
  CHECK_FALSE(machine.process_event(emel::batch::planner::event::plan_request{
      .token_ids = tokens.data(),
      .n_tokens = static_cast<int32_t>(tokens.size()),
      .n_steps = 1,
      .mode = static_cast<emel::batch::planner::event::plan_mode>(99),
      .equal_sequential = false,
      .on_done = make_done(&mode_capture),
      .on_error = make_error(&mode_capture),
  }));
  CHECK(mode_capture.error_called);
  CHECK(mode_capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_mode));
}
