#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::generator::frame {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  tokenize_failed = (1u << 1),
  planning_failed = (1u << 2),
  predict_failed = (1u << 3),
  graph_failed = (1u << 4),
  sample_failed = (1u << 5),
  detokenize_failed = (1u << 6),
  frame_pending = (1u << 7),
  unexpected_event = (1u << 8),
};

} // namespace emel::speech::generator::frame
