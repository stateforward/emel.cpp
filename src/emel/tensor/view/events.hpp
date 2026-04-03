#pragma once

#include <cstdint>

namespace emel::tensor::view::event {

template <class policy>
struct capture_tensor_view {
  typename policy::tensor_machine_type * tensor_machine = nullptr;
  int32_t tensor_id = 0;
  typename policy::tensor_state_type * state_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::tensor::view::event

namespace emel::tensor::view::events {

template <class policy>
struct capture_tensor_view_done {
  const event::capture_tensor_view<policy> * request = nullptr;
};

template <class policy>
struct capture_tensor_view_error {
  int32_t err = 0;
  const event::capture_tensor_view<policy> * request = nullptr;
};

}  // namespace emel::tensor::view::events
