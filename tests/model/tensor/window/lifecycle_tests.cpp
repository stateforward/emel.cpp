#include <doctest/doctest.h>

#include "../../../allocation_tracker.hpp"
#include "window_test_fixture.hpp"

// Lifecycle coverage for the tensor window machine: bind validation,
// passthrough vs streaming budget decision, explicit not_bound/not_streaming
// error legs, unbind teardown, and allocation-free acquire dispatch.

using namespace emel_window_test;

TEST_CASE("tensor window starts unbound and rejects acquire and unbind") {
  window_fixture fixture{};
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_unbound>));

  acquire_capture acquire{};
  CHECK_FALSE(fixture.acquire(0, acquire));
  CHECK(acquire.error);
  CHECK(acquire.err == emel::error::cast(window::error::not_bound));

  unbind_capture unbind{};
  CHECK_FALSE(fixture.unbind(unbind));
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_unbound>));
}

TEST_CASE("tensor window rejects invalid bind requests") {
  stream_file file{"invalid_bind"};
  window_fixture fixture{};
  bind_capture capture{};

  const window::event::bind_window_request bad_chunk{
      .file_path = file.path_str,
      .file_size_bytes = file.file_size,
      .extents = file.extents,
      .layer_weight_counts = file.layer_weight_counts,
      .budget_bytes = 0u,
      .window_slots = 4u,
      .prefetch_depth = 2u,
      .stage_chunk_bytes = 64u * 1024u,  // below k_min_stream_chunk_bytes
  };
  window::event::bind_window bind_request{bad_chunk};
  bind_request.on_error = {&capture, on_bind_error};
  CHECK_FALSE(fixture.machine.process_event(bind_request));
  CHECK(capture.error);
  CHECK(capture.err == emel::error::cast(window::error::invalid_request));
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_unbound>));
}

TEST_CASE("tensor window passthrough activates without slots for fitting budgets") {
  stream_file file{"passthrough"};
  window_fixture fixture{};
  bind_capture capture{};

  CHECK(fixture.bind(file, capture, /*budget=*/0u));
  CHECK(capture.done);
  CHECK_FALSE(capture.streaming_active);
  CHECK(capture.source_base != nullptr);
  CHECK(capture.window_slots == 0u);
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_passthrough_ready>));

  acquire_capture acquire{};
  CHECK_FALSE(fixture.acquire(0, acquire));
  CHECK(acquire.error);
  CHECK(acquire.err == emel::error::cast(window::error::not_streaming));

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
  CHECK(unbind.done);
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_unbound>));
}

TEST_CASE("tensor window streaming bind activates slots and reports source") {
  stream_file file{"stream_bind"};
  window_fixture fixture{};
  bind_capture capture{};

  CHECK(fixture.bind(file, capture, streaming_budget()));
  CHECK(capture.done);
  CHECK(capture.streaming_active);
  CHECK(capture.source_base != nullptr);
  CHECK(capture.window_slots == 4u);
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_ready>));

  acquire_capture out_of_range{};
  CHECK_FALSE(fixture.acquire(static_cast<int32_t>(k_layers), out_of_range));
  CHECK(out_of_range.error);
  CHECK(out_of_range.err ==
        emel::error::cast(window::error::layer_out_of_range));

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
  CHECK(unbind.done);
}

TEST_CASE("tensor window rejects rebind while bound") {
  stream_file file{"rebind"};
  window_fixture fixture{};
  bind_capture first{};
  CHECK(fixture.bind(file, first, streaming_budget()));

  bind_capture second{};
  CHECK_FALSE(fixture.bind(file, second, streaming_budget()));
  CHECK(second.error);
  CHECK(second.err == emel::error::cast(window::error::already_bound));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_ready>));

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window acquire dispatch does not allocate") {
  stream_file file{"no_alloc"};
  window_fixture fixture{};
  bind_capture capture{};
  CHECK(fixture.bind(file, capture, streaming_budget()));

  // Warm one full pass so every slot storage and ticket is exercised.
  for (uint32_t layer = 0; layer < k_layers; ++layer) {
    acquire_capture warm{};
    CHECK(fixture.acquire(static_cast<int32_t>(layer), warm));
  }

  emel::test::allocation::allocation_scope allocations{};
  for (uint32_t layer = 0; layer < k_layers; ++layer) {
    acquire_capture acquire{};
    CHECK(fixture.acquire(static_cast<int32_t>(layer), acquire));
    CHECK(acquire.done);
  }
  CHECK(allocations.allocations() == 0u);

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}
