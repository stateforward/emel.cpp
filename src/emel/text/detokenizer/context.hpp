#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::text::detokenizer::action {

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  bool is_bound = false;

  int32_t token_id = -1;
  bool emit_special = false;
  uint8_t * pending_bytes = nullptr;
  size_t pending_length = 0;
  size_t pending_capacity = 0;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t output_length = 0;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::text::detokenizer::action
