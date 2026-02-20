#pragma once

#include <cstdint>

#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::realloc_analyzer::action {

struct context {
  event::graph_view graph = {};
  const event::node_alloc * node_allocs = nullptr;
  int32_t node_alloc_count = 0;
  const event::leaf_alloc * leaf_allocs = nullptr;
  int32_t leaf_alloc_count = 0;
  bool needs_realloc = false;
  uint32_t step = 0;
  int32_t phase_error = EMEL_OK;
};

}  // namespace emel::buffer::realloc_analyzer::action
