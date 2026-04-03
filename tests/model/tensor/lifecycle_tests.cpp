#include <cstdint>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"

namespace {

emel::model::data::tensor_record make_tensor_record() {
  emel::model::data::tensor_record tensor{};
  tensor.type = 9;
  tensor.file_offset = 4096u;
  tensor.data_size = 512u;
  tensor.file_index = 3u;
  return tensor;
}

const void * fake_buffer(const uintptr_t value) {
  return reinterpret_cast<const void *>(value);
}

}  // namespace

TEST_CASE("model_tensor_bind_capture_evict_cycle") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));

  auto bind =
      emel::model::tensor::event::bind_tensor{12, tensor_record, fake_buffer(0x9000u), 512u};
  bind.error_out = &err;
  REQUIRE(machine.process_event(bind));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none)));

  emel::model::tensor::event::tensor_state state{};
  REQUIRE(machine.process_event(emel::model::tensor::event::capture_tensor_state{
    .tensor_id = 12,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == emel::model::tensor::event::lifecycle::resident);
  CHECK(state.buffer == fake_buffer(0x9000u));
  CHECK(state.buffer_bytes == 512u);
  CHECK(state.file_offset == 4096u);
  CHECK(state.data_size == 512u);
  CHECK(state.file_index == 3u);
  CHECK(state.tensor_type == 9);

  REQUIRE(machine.process_event(emel::model::tensor::event::evict_tensor{
    .tensor_id = 12,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(emel::model::tensor::event::capture_tensor_state{
    .tensor_id = 12,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == emel::model::tensor::event::lifecycle::evicted);
  CHECK(state.buffer == nullptr);
  CHECK(state.buffer_bytes == 0u);
  CHECK(state.file_offset == 4096u);
}

TEST_CASE("model_tensor_rejects_invalid_requests") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));
  emel::model::tensor::event::tensor_state state{};

  auto invalid_bind = emel::model::tensor::event::bind_tensor{0, tensor_record, nullptr, 512u};
  invalid_bind.error_out = &err;
  CHECK_FALSE(machine.process_event(invalid_bind));
  CHECK_FALSE(machine.process_event(emel::model::tensor::event::capture_tensor_state{
    .tensor_id = 0,
    .state_out = nullptr,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::invalid_request)));

  CHECK_FALSE(machine.process_event(emel::model::tensor::event::evict_tensor{
    .tensor_id = 5,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::invalid_request)));

  auto resident_bind =
      emel::model::tensor::event::bind_tensor{5, tensor_record, fake_buffer(0x9100u), 512u};
  resident_bind.error_out = &err;
  REQUIRE(machine.process_event(resident_bind));
  REQUIRE(machine.process_event(emel::model::tensor::event::capture_tensor_state{
    .tensor_id = 5,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == emel::model::tensor::event::lifecycle::resident);
}

TEST_CASE("model_tensor_unexpected_event_keeps_machine_dispatchable") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));

  CHECK(machine.process_event(emel::model::tensor::events::bind_tensor_done{}));
  auto bind =
      emel::model::tensor::event::bind_tensor{2, tensor_record, fake_buffer(0x9200u), 256u};
  bind.error_out = &err;
  CHECK(machine.process_event(bind));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none)));
}
