#pragma once

#include "parity_runner.hpp"

#include <string_view>

namespace emel::paritychecker {

struct engine_adapter {
  parity_mode mode = parity_mode::tokenizer;
  std::string_view name = {};
  int (*run)(const parity_options & opts) = nullptr;
};

const engine_adapter * find_engine(parity_mode mode);

}  // namespace emel::paritychecker
