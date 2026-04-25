#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/graph/tensor/sm.hpp"

struct docsgen_tensor_view_policy {
  using tensor_machine_type = emel::graph::tensor::sm;
  using tensor_state_type = emel::graph::tensor::event::tensor_state;
  static constexpr int32_t max_tensors = emel::graph::tensor::detail::max_tensors;
  static constexpr int32_t none_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::none));
  static constexpr int32_t invalid_request_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::invalid_request));
  static constexpr int32_t internal_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::internal_error));

  static bool capture_tensor_state(tensor_machine_type &,
                                   const int32_t,
                                   tensor_state_type &,
                                   int32_t & err_out) noexcept {
    err_out = none_error_code;
    return true;
  }
};
