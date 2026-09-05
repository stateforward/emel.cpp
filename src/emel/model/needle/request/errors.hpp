#pragma once

#include "emel/error/error.hpp"

namespace emel::model::needle::request {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  not_initialized = (1u << 1),
  capacity_exceeded = (1u << 2),
  tokenizer_rejected = (1u << 3),
  graph_rejected = (1u << 4),
  detokenizer_rejected = (1u << 5),
  response_invalid = (1u << 6),
  internal_error = (1u << 7),
};

} // namespace emel::model::needle::request
