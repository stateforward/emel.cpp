#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/parser/gguf/events.hpp"

namespace emel::parser::gguf::action {

struct context {
  requirements probed = {};
  void * kv_arena = nullptr;
  uint64_t kv_arena_size = 0;
  void * kv_entries = nullptr;
  uint32_t kv_entry_capacity = 0;
  emel::model::data::tensor_record * tensors = nullptr;
  uint32_t tensor_capacity = 0;
  int32_t last_error = EMEL_OK;
  bool probed_ok = false;
  bool bound_ok = false;
};

}  // namespace emel::parser::gguf::action
