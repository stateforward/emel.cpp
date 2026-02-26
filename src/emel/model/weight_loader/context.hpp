#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::model::weight_loader::action {

struct context {
  emel::model::data::tensor_record * tensors = nullptr;
  uint32_t tensor_count = 0;
  uint32_t planned_effects = 0;
  int32_t last_error = EMEL_OK;
  bool bound_ok = false;
};

}  // namespace emel::model::weight_loader::action
