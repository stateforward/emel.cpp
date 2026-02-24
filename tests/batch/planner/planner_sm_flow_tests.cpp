#include <algorithm>
#include <array>
#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/batch/planner/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"


namespace {

struct plan_capture {
  std::array<int32_t, 8> sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::planner::events::plan_done & ev) noexcept {
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

TEST_CASE("batch_planner_sm_recover_after_error") {
  emel::batch::planner::sm machine{};
  plan_capture error_capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&error_capture),
    .on_error = make_error(&error_capture),
  }));
  CHECK(error_capture.error_called);
  CHECK(machine.is(boost::sml::state<emel::batch::planner::invalid_request>));

  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  plan_capture ok_capture{};

  CHECK(machine.process_event(emel::batch::planner::event::plan{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&ok_capture),
    .on_error = make_error(&ok_capture),
  }));

  CHECK(ok_capture.done_called);
  CHECK(machine.is(boost::sml::state<emel::batch::planner::done>));
}

TEST_CASE("batch_planner_sm_accepts_consecutive_splits") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  plan_capture first{};
  plan_capture second{};

  CHECK(machine.process_event(emel::batch::planner::event::plan{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 2,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&first),
    .on_error = make_error(&first),
  }));
  CHECK(first.done_called);

  CHECK(machine.process_event(emel::batch::planner::event::plan{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&second),
    .on_error = make_error(&second),
  }));
  CHECK(second.done_called);
}
