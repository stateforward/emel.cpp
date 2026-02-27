#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::model {
struct data;
}

namespace emel::model::architecture {

using map_layers_fn = emel::callback<emel::error::type(emel::model::data &)>;
using validate_fn = emel::callback<emel::error::type(const emel::model::data &)>;

struct type {
  std::string_view architecture_name = {};
  map_layers_fn map_layers = {};
  validate_fn validate = {};
};

}  // namespace emel::model::architecture
