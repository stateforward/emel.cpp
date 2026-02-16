#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/planner/sm.hpp"

namespace {

bool planner_done_callback(void *, const emel::buffer::planner::events::plan_done &) {
  return true;
}

bool planner_error_callback(void *, const emel::buffer::planner::events::plan_error &) {
  return true;
}

TEST_CASE("buffer_planner_sm_successful_plan_dispatches_done") {
  emel::buffer::planner::sm machine{};
  int32_t err = EMEL_OK;
  int32_t sizes_out = 0;

  CHECK(machine.process_event(emel::buffer::planner::event::plan{
    .graph = {},
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = &sizes_out,
    .sizes_out_count = 1,
    .error_out = &err,
    .owner_sm = reinterpret_cast<void *>(0x1),
    .dispatch_done = planner_done_callback,
    .dispatch_error = planner_error_callback,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_planner_sm_rejects_invalid_plan") {
  emel::buffer::planner::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::buffer::planner::event::plan{
    .graph = {},
    .buffer_count = 1,
    .size_only = false,
    .error_out = &err,
    .owner_sm = nullptr,
    .dispatch_done = nullptr,
    .dispatch_error = nullptr,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
