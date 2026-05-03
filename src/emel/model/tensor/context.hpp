#pragma once

#include <span>

#include "emel/model/data.hpp"
#include "emel/model/tensor/detail.hpp"

namespace emel::model::tensor::action {

struct context {
  detail::tensor_storage tensors = {};
  std::span<emel::model::data::tensor_record> bound_records = {};
};

} // namespace emel::model::tensor::action
