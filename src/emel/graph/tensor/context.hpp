#pragma once

#include "emel/graph/tensor/detail.hpp"

namespace emel::graph::tensor::action {

struct context {
  detail::tensor_pool tensors = {};
};

}  // namespace emel::graph::tensor::action
