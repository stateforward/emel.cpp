#pragma once

#include "emel/tensor/detail.hpp"

namespace emel::tensor::action {

struct context {
  detail::tensor_pool tensors = {};
};

}  // namespace emel::tensor::action
