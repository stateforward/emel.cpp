#pragma once

#include <span>

#include "emel/gguf/loader/events.hpp"
#include "emel/model/data.hpp"

namespace emel::gguf::loader::action {

struct context {
  requirements probed = {};
  std::span<uint8_t> kv_arena = {};
  std::span<kv_entry> kv_entries = {};
  std::span<emel::model::data::tensor_record> tensors = {};
};

}  // namespace emel::gguf::loader::action
