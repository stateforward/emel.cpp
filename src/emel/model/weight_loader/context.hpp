#pragma once

#include <cstdint>

#include "emel/model/data.hpp"

namespace emel::model::weight_loader::action {

struct context {
  emel::model::data::tensor_record * tensors = nullptr;
  uint32_t tensor_count = 0u;
  uint32_t planned_effects = 0u;
};

}  // namespace emel::model::weight_loader::action
