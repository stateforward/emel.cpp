#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/planner/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"


namespace {

struct plan_capture {
  int32_t err = emel::error::cast(emel::batch::planner::error::none);
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::planner::events::plan_done &) noexcept {
    done_called = true;
    err = emel::error::cast(emel::batch::planner::error::none);
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

TEST_CASE("batch_planner_sm_successful_split") {
  emel::batch::planner::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  plan_capture capture{};

  machine.process_event(emel::batch::planner::event::plan_request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_steps = 1,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  });

  CHECK(capture.done_called);
  CHECK(machine.is(boost::sml::state<emel::batch::planner::state_completed>));
}

TEST_CASE("batch_planner_sm_validation_error_path") {
  emel::batch::planner::sm machine{};
  plan_capture capture{};

  machine.process_event(emel::batch::planner::event::plan_request{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_steps = 1,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  });

  CHECK(capture.error_called);
  CHECK(capture.err == emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_token_data));
  CHECK(machine.is(boost::sml::state<emel::batch::planner::state_request_rejected>));
}
