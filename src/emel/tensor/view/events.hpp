#pragma once

#include <cstdint>

namespace emel::tensor {
struct sm;
}  // namespace emel::tensor

namespace emel::tensor::event {
struct tensor_state;
}  // namespace emel::tensor::event

namespace emel::tensor::view::event {

struct capture_tensor_view {
  emel::tensor::sm * tensor_machine = nullptr;
  int32_t tensor_id = 0;
  emel::tensor::event::tensor_state * state_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::tensor::view::event

namespace emel::tensor::view::events {

struct capture_tensor_view_done {
  const event::capture_tensor_view * request = nullptr;
};

struct capture_tensor_view_error {
  int32_t err = 0;
  const event::capture_tensor_view * request = nullptr;
};

}  // namespace emel::tensor::view::events
