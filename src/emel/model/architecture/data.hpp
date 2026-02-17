#pragma once

#include <cstdint>

namespace emel::model {
struct data;
}

namespace emel::model::architecture {



using map_layers_fn = void (*)(emel::model::data &);
using validate_fn = bool (*)(const emel::model::data &);

struct data {
  const char * architecture_name = nullptr;
  map_layers_fn map_layers = nullptr;
  validate_fn validate = nullptr;
};

}  // namespace emel::model::architecture
