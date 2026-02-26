#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/parser/gguf/events.hpp"

namespace emel::model::loader::action {

struct context {
  const event::load * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::parser::gguf::requirements parser_requirements = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::model::loader::action
