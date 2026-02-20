#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/parser/dispatch.hpp"

namespace emel::model::loader::action {

struct context {
  const event::load * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::parser::kind parser_kind = emel::parser::kind::count;
  void * parser_sm = nullptr;
  emel::parser::dispatch_parse_fn parser_dispatch = nullptr;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::model::loader::action
