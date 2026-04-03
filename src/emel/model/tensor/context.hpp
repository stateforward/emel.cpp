#pragma once

#include "emel/model/tensor/detail.hpp"

namespace emel::model::tensor::action {

struct context {
  detail::tensor_storage tensors = {};
};

}  // namespace emel::model::tensor::action
