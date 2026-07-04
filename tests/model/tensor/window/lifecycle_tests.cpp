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

TEST_CASE("tensor window rejects extents outside the mapped source") {
  stream_file file{"bad_extent"};
  window_fixture fixture{};
  bind_capture capture{};

  // Copy the fixture extents and push one past the end of the file.
  std::vector<window::detail::weight_extent> extents{file.extents.begin(),
                                                     file.extents.end()};
  extents.back().file_offset = file.file_size - 16u;
  extents.back().byte_size = 64u;

  const window::event::bind_window_request request{
      .file_path = file.path_str,
      .file_size_bytes = file.file_size,
      .extents = extents,
      .layer_weight_counts = file.layer_weight_counts,
      .budget_bytes = 0u,
      .window_slots = 4u,
      .prefetch_depth = 2u,
      .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
  };
  window::event::bind_window bind_request{request};
  bind_request.on_error = {&capture, on_bind_error};
  CHECK_FALSE(fixture.machine.process_event(bind_request));
  CHECK(capture.error);
  CHECK(capture.err == emel::error::cast(window::error::invalid_request));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));
}

TEST_CASE("tensor window passthrough accepts zero slot fields") {
  stream_file file{"passthrough_zero_slots"};
  window_fixture fixture{};
  bind_capture capture{};

  // A passthrough bind never allocates or uses slots, so the documented
  // zero defaults for window_slots/prefetch_depth must be accepted.
  CHECK(fixture.bind(file, capture, /*budget=*/0u, /*slots=*/0u,
                     /*prefetch_depth=*/0u));
  CHECK(capture.done);
  CHECK_FALSE(capture.streaming_active);
  CHECK(fixture.machine.is(
      stateforward::sml::state<window::state_passthrough_ready>));

  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window handles absent optional callbacks without crashing") {
  stream_file file{"no_callbacks"};

  {
    // bind without on_done: success is still reported via the return value.
    window_fixture fixture{};
    const window::event::bind_window_request request{
        .file_path = file.path_str,
        .file_size_bytes = file.file_size,
        .extents = file.extents,
        .layer_weight_counts = file.layer_weight_counts,
        .budget_bytes = 0u,
        .window_slots = 0u,
        .prefetch_depth = 0u,
        .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
    };
    window::event::bind_window bind_request{request};
    CHECK(fixture.machine.process_event(bind_request));
    CHECK(fixture.machine.is(
        stateforward::sml::state<window::state_passthrough_ready>));
  }

  {
    // acquire on an unbound machine without an error callback: the dispatch
    // fails without dereferencing the empty callback.
    window_fixture fixture{};
    window::event::acquire_layer_window request{0};
    CHECK_FALSE(fixture.machine.process_event(request));
    CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));
  }

  {
    // streaming bind without on_done still activates and primes.
    window_fixture fixture{};
    const window::event::bind_window_request request{
        .file_path = file.path_str,
        .file_size_bytes = file.file_size,
        .extents = file.extents,
        .layer_weight_counts = file.layer_weight_counts,
        .budget_bytes = streaming_budget(),
        .slot_storage = std::span<uint8_t>{fixture.slot_arena},
        .window_slots = 4u,
        .prefetch_depth = 2u,
        .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
    };
    window::event::bind_window bind_request{request};
    CHECK(fixture.machine.process_event(bind_request));
    CHECK(fixture.machine.is(stateforward::sml::state<window::state_ready>));
    unbind_capture unbind{};
    CHECK(fixture.unbind(unbind));
  }
}

TEST_CASE("tensor window rejects invalid streaming slot configuration") {
  stream_file file{"bad_slots"};
  window_fixture fixture{};
  bind_capture capture{};

  // Budget forces streaming, but a single slot cannot form a prefetch ring:
  // the streaming-config guard routes invalid_request and the mapping is
  // released (verified by the following well-formed bind).
  CHECK_FALSE(fixture.bind(file, capture, streaming_budget(), /*slots=*/1u,
                           /*prefetch_depth=*/0u));
  CHECK(capture.error);
  CHECK(capture.err == emel::error::cast(window::error::invalid_request));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));

  bind_capture retry{};
  CHECK(fixture.bind(file, retry, streaming_budget()));
  CHECK(retry.done);
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window failure guards and effects route their errors") {
  // Direct functor checks for the routes that need runtime states the fixture
  // cannot force with real collaborators (a prefetch-failed slot, the
  // failed-release return modes).
  window_fixture fixture{};
  window::action::context ctx = fixture.make_context();

  window::detail::stream_scheduler scheduler{};

  // A slot committed as failed for the requested layer routes the resubmit
  // guard, and only that guard.
  ctx.window.layer_count = 4u;
  ctx.window.slot_count = 2u;
  ctx.window.slots[0].layer = 0;
  ctx.window.slots[0].lifecycle = window::detail::slot_lifecycle::failed;
  const window::event::acquire_layer_window acquire{0};
  window::detail::acquire_attempt_status acquire_status{};
  window::detail::acquire_runtime acquire_runtime{acquire, acquire_status,
                                                  scheduler};
  CHECK(window::guard::guard_acquire_layer_failed{}(acquire_runtime, ctx));
  CHECK_FALSE(
      window::guard::guard_acquire_layer_resident{}(acquire_runtime, ctx));
  CHECK_FALSE(
      window::guard::guard_acquire_layer_loading{}(acquire_runtime, ctx));
  CHECK_FALSE(
      window::guard::guard_acquire_layer_unscheduled{}(acquire_runtime, ctx));

  // Out-of-range layers reject through the shared range check.
  const window::event::acquire_layer_window out_of_range{99};
  window::detail::acquire_attempt_status range_status{};
  window::detail::acquire_runtime range_runtime{out_of_range, range_status,
                                                scheduler};
  CHECK_FALSE(window::guard::guard_acquire_layer_failed{}(range_runtime, ctx));

  // The failed-release exits return to the mode the window is still in.
  const window::event::unbind_window unbind_request{};
  window::detail::unbind_attempt_status unbind_status{};
  const window::detail::unbind_finish_runtime unbind_runtime{unbind_request,
                                                             unbind_status};
  ctx.window.streaming_active = true;
  CHECK(window::guard::guard_unbind_window_streaming{}(unbind_runtime, ctx));
  CHECK_FALSE(
      window::guard::guard_unbind_window_passthrough{}(unbind_runtime, ctx));
  CHECK(window::guard::guard_unbind_error_callback_absent_streaming{}(
      unbind_runtime, ctx));
  CHECK_FALSE(window::guard::guard_unbind_error_callback_absent_passthrough{}(
      unbind_runtime, ctx));
  CHECK_FALSE(window::guard::guard_unbind_error_callback_present_streaming{}(
      unbind_runtime, ctx));
  ctx.window.streaming_active = false;
  CHECK(window::guard::guard_unbind_window_passthrough{}(unbind_runtime, ctx));
  CHECK(window::guard::guard_unbind_error_callback_absent_passthrough{}(
      unbind_runtime, ctx));
  CHECK_FALSE(window::guard::guard_unbind_error_callback_present_passthrough{}(
      unbind_runtime, ctx));

  // With an error callback present the present-mode composes select instead.
  unbind_capture unbind_owner{};
  window::event::unbind_window unbind_with_callback{};
  unbind_with_callback.on_error = {&unbind_owner, on_unbind_error};
  window::detail::unbind_attempt_status callback_status{};
  const window::detail::unbind_finish_runtime callback_runtime{
      unbind_with_callback, callback_status};
  CHECK(window::guard::guard_unbind_error_callback_present_passthrough{}(
      callback_runtime, ctx));
  CHECK_FALSE(window::guard::guard_unbind_error_callback_absent_passthrough{}(
      callback_runtime, ctx));
  ctx.window.streaming_active = true;
  CHECK(window::guard::guard_unbind_error_callback_present_streaming{}(
      callback_runtime, ctx));
  CHECK_FALSE(window::guard::guard_unbind_error_callback_absent_streaming{}(
      callback_runtime, ctx));

  // Reset the crafted slot state so the context destructor sees no storage.
  ctx.window = {};
}

TEST_CASE("tensor window rejection without callbacks reports via return value") {
  stream_file file{"reject_no_callbacks"};
  window_fixture fixture{};

  // A post-map rejection with no callbacks takes the absent exit routes
  // (release outcome composed) and reports failure via the return value.
  const window::event::bind_window_request request{
      .file_path = file.path_str,
      .file_size_bytes = file.file_size,
      .extents = file.extents,
      .layer_weight_counts = file.layer_weight_counts,
      .budget_bytes = 2u * k_weight_bytes,  // forces streaming, too small
      .slot_storage = std::span<uint8_t>{fixture.slot_arena},
      .window_slots = 4u,
      .prefetch_depth = 2u,
      .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
  };
  window::event::bind_window bind_request{request};
  CHECK_FALSE(fixture.machine.process_event(bind_request));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));

  bind_capture retry{};
  CHECK(fixture.bind(file, retry, streaming_budget()));
  CHECK(retry.done);
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window rejects undersized slot storage explicitly") {
  stream_file file{"small_arena"};
  window_fixture fixture{};
  bind_capture capture{};

  // A valid streaming config whose caller arena cannot hold the slots must
  // report slot_storage_too_small (and release the mapping, proven by the
  // follow-up bind).
  alignas(
      window::detail::k_slot_alignment_bytes) static std::array<uint8_t, 64u>
      tiny_arena{};
  CHECK_FALSE(fixture.bind(file, capture, streaming_budget(), 4u, 2u,
                           std::span<uint8_t>{tiny_arena},
                           /*use_override=*/true));
  CHECK(capture.error);
  CHECK(capture.err ==
        emel::error::cast(window::error::slot_storage_too_small));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));

  bind_capture retry{};
  CHECK(fixture.bind(file, retry, streaming_budget()));
  CHECK(retry.done);
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window rejects null slot storage explicitly") {
  stream_file file{"null_arena"};
  window_fixture fixture{};
  bind_capture capture{};

  // A null base is 64-byte aligned and can carry any span length, so the
  // storage guard must reject it before the slot partition derives worker
  // destinations from address zero.
  CHECK_FALSE(fixture.bind(file, capture, streaming_budget(), 4u, 2u,
                           std::span<uint8_t>{static_cast<uint8_t *>(nullptr),
                                              size_t{1} << 20},
                           /*use_override=*/true));
  CHECK(capture.error);
  CHECK(capture.err ==
        emel::error::cast(window::error::slot_storage_too_small));
  CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));

  bind_capture retry{};
  CHECK(fixture.bind(file, retry, streaming_budget()));
  CHECK(retry.done);
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
}

TEST_CASE("tensor window releases the source mapping on rejected binds") {
  stream_file file{"budget_reject"};
  window_fixture fixture{};

  // Each rejected bind must release its source mapping back to the io_mmap
  // pool; a leak would exhaust the fixed mapping slots across iterations.
  const uint64_t layer_bytes = 2u * k_weight_bytes;
  for (uint32_t attempt = 0; attempt < 3u; ++attempt) {
    bind_capture capture{};
    // budget below the 4-slot working set but nonzero: budget_too_small.
    CHECK_FALSE(fixture.bind(file, capture, /*budget=*/layer_bytes, 4u, 2u));
    CHECK(capture.error);
    CHECK(capture.err == emel::error::cast(window::error::budget_too_small));
    CHECK(fixture.machine.is(stateforward::sml::state<window::state_unbound>));
  }

  // A well-formed bind after the rejections succeeds with a fresh mapping.
  bind_capture capture{};
  CHECK(fixture.bind(file, capture, streaming_budget()));
  CHECK(capture.done);
  CHECK(capture.streaming_active);
  unbind_capture unbind{};
  CHECK(fixture.unbind(unbind));
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
