#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::text::detokenizer::action {

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  bool is_bound = false;
};

}  // namespace emel::text::detokenizer::action
