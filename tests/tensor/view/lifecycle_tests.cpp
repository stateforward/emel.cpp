#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/events.hpp"
#include "emel/tensor/sm.hpp"
#include "emel/tensor/view/events.hpp"
#include "emel/tensor/view/sm.hpp"

namespace {

using tensor_sm = emel::tensor::sm;
using tensor_view_sm = emel::tensor::view::sm;

void * fake_buffer(const uintptr_t value) {
  return reinterpret_cast<void *>(value);
}

}  // namespace

TEST_CASE("tensor_view_capture_tensor_view_reads_tensor_state") {
  tensor_sm tensors{};
  tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  REQUIRE(tensors.process_event(emel::tensor::event::reserve_tensor{
    .tensor_id = 21,
    .buffer = fake_buffer(0x7000u),
    .buffer_bytes = 96u,
    .consumer_refs = 2,
    .is_leaf = false,
    .error_out = &err,
  }));
  REQUIRE(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));

  emel::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = &tensors,
    .tensor_id = 21,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
  CHECK(state.lifecycle_state == emel::tensor::event::lifecycle::empty);
  CHECK(state.seed_refs == 2u);
  CHECK(state.live_refs == 2u);
}

TEST_CASE("tensor_view_capture_tensor_view_rejects_invalid_request") {
  tensor_sm tensors{};
  tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::tensor::event::tensor_state state{};

  CHECK_FALSE(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = nullptr,
    .tensor_id = 0,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));

  CHECK_FALSE(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = &tensors,
    .tensor_id = -1,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));

  CHECK_FALSE(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = &tensors,
    .tensor_id = 0,
    .state_out = nullptr,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));
}

TEST_CASE("tensor_view_capture_tensor_view_propagates_tensor_error") {
  tensor_sm tensors{};
  tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  REQUIRE(tensors.process_event(emel::tensor::event::reserve_tensor{
    .tensor_id = 31,
    .buffer = fake_buffer(0x7100u),
    .buffer_bytes = 128u,
    .consumer_refs = 1,
    .is_leaf = true,
    .error_out = &err,
  }));
  CHECK_FALSE(tensors.process_event(emel::tensor::event::publish_filled_tensor{
    .tensor_id = 31,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::error::internal_error)));

  emel::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = &tensors,
    .tensor_id = 31,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
  CHECK(state.lifecycle_state == emel::tensor::event::lifecycle::internal_error);
}

TEST_CASE("tensor_view_unexpected_event_keeps_machine_dispatchable") {
  tensor_sm tensors{};
  tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::tensor::event::tensor_state state{};

  CHECK(view.process_event(emel::tensor::view::events::capture_tensor_view_done{}));

  CHECK(view.process_event(emel::tensor::view::event::capture_tensor_view{
    .tensor_machine = &tensors,
    .tensor_id = 0,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
}
