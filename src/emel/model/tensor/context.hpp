#pragma once

#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/model/tensor/detail.hpp"

namespace emel::model::tensor::action {

struct context {
  detail::tensor_storage tensors = {};
  uint32_t bound_count = 0u;
};

} // namespace emel::model::tensor::action
