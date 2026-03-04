#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/events.hpp"
#include "emel/tensor/sm.hpp"

namespace {

using tensor_sm = emel::tensor::sm;
using namespace emel::tensor;

void * fake_buffer(const uintptr_t value) {
  return reinterpret_cast<void *>(value);
}

}  // namespace

TEST_CASE("tensor_lifecycle_compute_publish_release_cycle") {
  tensor_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::error::none));

  REQUIRE(machine.process_event(event::reserve_tensor{
    .tensor_id = 7,
    .buffer = fake_buffer(0x1000u),
    .buffer_bytes = 256u,
    .consumer_refs = 2,
    .is_leaf = false,
    .error_out = &err,
  }));
  REQUIRE(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::none)));

  event::tensor_state state{};
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 7,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::empty);
  CHECK(state.is_leaf == 0u);
  CHECK(state.seed_refs == 2u);
  CHECK(state.live_refs == 2u);

  REQUIRE(machine.process_event(event::publish_filled_tensor{
    .tensor_id = 7,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 7,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::filled);
  CHECK(state.live_refs == 2u);

  REQUIRE(machine.process_event(event::release_tensor_ref{
    .tensor_id = 7,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 7,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::filled);
  CHECK(state.live_refs == 1u);

  REQUIRE(machine.process_event(event::release_tensor_ref{
    .tensor_id = 7,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 7,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::empty);
  CHECK(state.live_refs == 0u);
}

TEST_CASE("tensor_lifecycle_leaf_reset_and_release_are_noops") {
  tensor_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::error::none));

  REQUIRE(machine.process_event(event::reserve_tensor{
    .tensor_id = 3,
    .buffer = fake_buffer(0x2000u),
    .buffer_bytes = 128u,
    .consumer_refs = 4,
    .is_leaf = true,
    .error_out = &err,
  }));

  event::tensor_state state{};
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 3,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::leaf_filled);
  CHECK(state.is_leaf == 1u);
  CHECK(state.seed_refs == 0u);
  CHECK(state.live_refs == 0u);

  REQUIRE(machine.process_event(event::release_tensor_ref{
    .tensor_id = 3,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::reset_tensor_epoch{
    .tensor_id = 3,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 3,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::leaf_filled);

  CHECK_FALSE(machine.process_event(event::publish_filled_tensor{
    .tensor_id = 3,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::internal_error)));
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 3,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::internal_error);
}

TEST_CASE("tensor_lifecycle_invalid_request_and_invalid_transition") {
  tensor_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::error::none));

  CHECK_FALSE(machine.process_event(event::reserve_tensor{
    .tensor_id = 0,
    .buffer = nullptr,
    .buffer_bytes = 64u,
    .consumer_refs = 1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::invalid_request)));

  REQUIRE(machine.process_event(event::reserve_tensor{
    .tensor_id = 0,
    .buffer = fake_buffer(0x3000u),
    .buffer_bytes = 64u,
    .consumer_refs = 1,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::publish_filled_tensor{
    .tensor_id = -1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 0,
    .state_out = nullptr,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::reset_tensor_epoch{
    .tensor_id = 1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::internal_error)));
}

TEST_CASE("tensor_lifecycle_reset_epoch_transitions_filled_to_empty") {
  tensor_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::error::none));

  REQUIRE(machine.process_event(event::reserve_tensor{
    .tensor_id = 11,
    .buffer = fake_buffer(0x4000u),
    .buffer_bytes = 512u,
    .consumer_refs = 1,
    .is_leaf = false,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::publish_filled_tensor{
    .tensor_id = 11,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::reset_tensor_epoch{
    .tensor_id = 11,
    .error_out = &err,
  }));

  event::tensor_state state{};
  REQUIRE(machine.process_event(event::capture_tensor_state{
    .tensor_id = 11,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == event::lifecycle::empty);
  CHECK(state.live_refs == 0u);
  CHECK(state.seed_refs == 1u);
}

TEST_CASE("tensor_lifecycle_unexpected_event_keeps_machine_dispatchable") {
  tensor_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::error::none));

  CHECK(machine.process_event(events::publish_filled_tensor_done{}));

  CHECK(machine.process_event(event::reserve_tensor{
    .tensor_id = 2,
    .buffer = fake_buffer(0x5000u),
    .buffer_bytes = 32u,
    .consumer_refs = 1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::none)));
}
