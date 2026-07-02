#include <doctest/doctest.h>

#include "window_test_fixture.hpp"

// Streaming behavior coverage: miss -> suspend -> commit content integrity,
// prefetch fast path, ring eviction across passes, and unbind joining
// in-flight slot loads.

using namespace emel_window_test;

TEST_CASE("tensor window streams every layer with correct content") {
  stream_file file{"content"};
  window_fixture fixture{};
  bind_capture capture{};
  REQUIRE(fixture.bind(file, capture, streaming_budget()));
  REQUIRE(capture.streaming_active);

  for (uint32_t layer = 0; layer < k_layers; ++layer) {
    acquire_capture acquire{};
    CHECK(fixture.acquire(static_cast<int32_t>(layer), acquire));
    CHECK(acquire.done);
    CHECK(slot_content_matches(acquire, layer));
  }

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window ring eviction preserves content across passes") {
  stream_file file{"eviction"};
  window_fixture fixture{};
  bind_capture capture{};
  REQUIRE(fixture.bind(file, capture, streaming_budget()));

  // Two full passes: with 4 slots over 6 layers every slot is evicted and
  // refilled at least once; the second pass re-streams evicted layers.
  for (uint32_t pass = 0; pass < 2u; ++pass) {
    for (uint32_t layer = 0; layer < k_layers; ++layer) {
      acquire_capture acquire{};
      CHECK(fixture.acquire(static_cast<int32_t>(layer), acquire));
      CHECK(slot_content_matches(acquire, layer));
    }
  }

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window cold acquire far from the prefetch ring streams correctly") {
  stream_file file{"cold_jump"};
  window_fixture fixture{};
  bind_capture capture{};
  REQUIRE(fixture.bind(file, capture, streaming_budget()));

  // Layer 5 is beyond the primed prefetch window: unscheduled miss that
  // submits, suspends the dispatch, and commits before returning.
  acquire_capture cold{};
  CHECK(fixture.acquire(5, cold));
  CHECK(cold.done);
  CHECK(slot_content_matches(cold, 5u));

  // Jumping back re-streams layer 0 through its (possibly evicted) slot.
  acquire_capture back{};
  CHECK(fixture.acquire(0, back));
  CHECK(slot_content_matches(back, 0u));

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window unbind joins in-flight prefetch loads") {
  stream_file file{"unbind_inflight"};
  window_fixture fixture{};
  bind_capture capture{};
  REQUIRE(fixture.bind(file, capture, streaming_budget()));

  // Prime submitted prefetch loads at bind; unbind immediately so the drain
  // must join whatever is still in flight before teardown.
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
  CHECK(unbind.done);
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_unbound>));

  // The source mapping was released: a fresh bind of the same file works.
  bind_capture rebind{};
  CHECK(fixture.bind(file, rebind, streaming_budget()));
  acquire_capture acquire{};
  CHECK(fixture.acquire(0, acquire));
  CHECK(slot_content_matches(acquire, 0u));
  unbind_capture unbind_again{};
  CHECK(fixture.unbind(unbind_again));
}
