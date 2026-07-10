#include <cstdint>
#include <limits>

#include <doctest/doctest.h>

#include "../../allocation_tracker.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/streaming/guards.hpp"
#include "emel/memory/streaming/sm.hpp"

namespace {

namespace streaming = emel::memory::streaming;
namespace sml = stateforward::sml;

int32_t error_code(const streaming::error value) noexcept {
  return static_cast<int32_t>(emel::error::cast(value));
}

struct fixture {
  explicit fixture(const int32_t capacity)
      : machine(streaming::dependencies{.capacity = capacity}) {}

  bool initialize() {
    err = error_code(streaming::error::none);
    return machine.process_event(
        streaming::event::initialize{.error_out = err});
  }

  bool advance() {
    err = error_code(streaming::error::none);
    return machine.process_event(
        streaming::event::advance{.result = result, .error_out = err});
  }

  bool reset() {
    err = error_code(streaming::error::none);
    return machine.process_event(streaming::event::reset{.error_out = err});
  }

  bool capture() {
    err = error_code(streaming::error::none);
    return machine.process_event(
        streaming::event::capture_view{.view_out = view, .error_out = err});
  }

  streaming::sm machine;
  streaming::advance_result result = {};
  streaming::window_view view = {};
  int32_t err = error_code(streaming::error::none);
};

void check_window(const streaming::window_view &view,
                  const int64_t logical_begin, const int64_t logical_end,
                  const int32_t physical_begin, const int32_t next_physical,
                  const int32_t valid_positions, const int32_t capacity) {
  CHECK(view.logical_begin == logical_begin);
  CHECK(view.logical_end == logical_end);
  CHECK(view.physical_begin == physical_begin);
  CHECK(view.next_physical_position == next_physical);
  CHECK(view.valid_positions == valid_positions);
  CHECK(view.capacity == capacity);
}

void check_advance(const fixture &state, const int64_t logical_position,
                   const int32_t physical_position, const int64_t logical_begin,
                   const int64_t logical_end, const int32_t physical_begin,
                   const int32_t next_physical, const int32_t valid_positions,
                   const int32_t capacity) {
  CHECK(state.result.logical_position == logical_position);
  CHECK(state.result.physical_position == physical_position);
  check_window(state.result.window, logical_begin, logical_end, physical_begin,
               next_physical, valid_positions, capacity);
}

} // namespace

TEST_CASE("memory streaming rejects invalid injected capacity") {
  fixture state{0};

  CHECK(state.initialize());
  CHECK(state.err == error_code(streaming::error::invalid_configuration));
  CHECK(state.machine.is(sml::state<streaming::state_uninitialized>));
}

TEST_CASE("memory streaming fills then wraps a fixed ring") {
  fixture state{3};

  REQUIRE(state.initialize());
  CHECK(state.machine.is(sml::state<streaming::state_empty>));
  REQUIRE(state.capture());
  check_window(state.view, 0, 0, 0, 0, 0, 3);

  REQUIRE(state.advance());
  check_advance(state, 0, 0, 0, 1, 0, 1, 1, 3);
  CHECK(state.machine.is(sml::state<streaming::state_filling>));

  REQUIRE(state.advance());
  check_advance(state, 1, 1, 0, 2, 0, 2, 2, 3);
  CHECK(state.machine.is(sml::state<streaming::state_filling>));
  REQUIRE(state.capture());
  check_window(state.view, 0, 2, 0, 2, 2, 3);

  REQUIRE(state.advance());
  check_advance(state, 2, 2, 0, 3, 0, 0, 3, 3);
  CHECK(state.machine.is(sml::state<streaming::state_full>));

  REQUIRE(state.advance());
  check_advance(state, 3, 0, 1, 4, 1, 1, 3, 3);

  REQUIRE(state.advance());
  check_advance(state, 4, 1, 2, 5, 2, 2, 3, 3);

  REQUIRE(state.advance());
  check_advance(state, 5, 2, 3, 6, 0, 0, 3, 3);

  REQUIRE(state.capture());
  check_window(state.view, 3, 6, 0, 0, 3, 3);
}

TEST_CASE("memory streaming capacity one stays full and overwrites one slot") {
  fixture state{1};

  REQUIRE(state.initialize());
  REQUIRE(state.advance());
  check_advance(state, 0, 0, 0, 1, 0, 0, 1, 1);
  CHECK(state.machine.is(sml::state<streaming::state_full>));

  REQUIRE(state.advance());
  check_advance(state, 1, 0, 1, 2, 0, 0, 1, 1);
}

TEST_CASE(
    "memory streaming reset invalidates history without clearing storage") {
  fixture state{2};

  REQUIRE(state.initialize());
  REQUIRE(state.advance());
  REQUIRE(state.advance());
  REQUIRE(state.advance());
  REQUIRE(state.reset());
  CHECK(state.machine.is(sml::state<streaming::state_empty>));

  REQUIRE(state.capture());
  check_window(state.view, 0, 0, 0, 0, 0, 2);

  REQUIRE(state.advance());
  check_advance(state, 0, 0, 0, 1, 0, 1, 1, 2);
}

TEST_CASE(
    "memory streaming models uninitialized and duplicate initialize errors") {
  fixture state{4};

  CHECK(state.advance());
  CHECK(state.err == error_code(streaming::error::uninitialized));
  CHECK(state.reset());
  CHECK(state.err == error_code(streaming::error::uninitialized));
  CHECK(state.capture());
  CHECK(state.err == error_code(streaming::error::uninitialized));

  REQUIRE(state.initialize());
  CHECK(state.initialize());
  CHECK(state.err == error_code(streaming::error::already_initialized));
  CHECK(state.machine.is(sml::state<streaming::state_empty>));
}

TEST_CASE("memory streaming dispatch remains allocation free") {
  fixture state{8};
  REQUIRE(state.initialize());

  emel::test::allocation::allocation_scope allocations;
  for (int32_t index = 0; index < 64; ++index) {
    REQUIRE(state.advance());
  }
  REQUIRE(state.capture());
  REQUIRE(state.reset());

  CHECK(allocations.allocations() == 0u);
}

TEST_CASE(
    "memory streaming makes unexpected events observable and recoverable") {
  struct unknown_event {};

  fixture state{3};
  REQUIRE(state.initialize());
  CHECK(state.machine.process_event(unknown_event{}));
  CHECK(state.machine.is(sml::state<streaming::state_errored>));

  CHECK(state.capture());
  CHECK(state.err == error_code(streaming::error::internal_error));
  REQUIRE(state.initialize());
  CHECK(state.machine.is(sml::state<streaming::state_empty>));
}

TEST_CASE("memory streaming full-state invariant failures are explicit") {
  streaming::action::context ctx{streaming::dependencies{.capacity = 3}};
  stateforward::sml::sm<streaming::model, stateforward::sml::testing> machine{
      ctx};
  machine.set_current_states(sml::state<streaming::state_full>);

  streaming::advance_result result = {};
  int32_t err = error_code(streaming::error::none);

  ctx.next_logical_position = std::numeric_limits<int64_t>::max();
  ctx.next_physical_position = 0;
  CHECK(machine.process_event(
      streaming::event::advance{.result = result, .error_out = err}));
  CHECK(err == error_code(streaming::error::position_overflow));

  ctx.next_logical_position = 3;
  ctx.next_physical_position = -1;
  err = error_code(streaming::error::none);
  CHECK(machine.process_event(
      streaming::event::advance{.result = result, .error_out = err}));
  CHECK(err == error_code(streaming::error::internal_error));

  ctx.next_physical_position = std::numeric_limits<int32_t>::max();
  err = error_code(streaming::error::none);
  CHECK(machine.process_event(
      streaming::event::advance{.result = result, .error_out = err}));
  CHECK(err == error_code(streaming::error::internal_error));

  const streaming::guard::guard_full_cursor_invalid cursor_invalid{};
  ctx.next_logical_position = std::numeric_limits<int64_t>::max();
  ctx.next_physical_position = 0;
  CHECK_FALSE(cursor_invalid(ctx));

  ctx.next_logical_position = 3;
  CHECK_FALSE(cursor_invalid(ctx));
}
