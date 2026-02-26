#pragma once

#include <cstdint>

#include "emel/graph/allocator/sm.hpp"

namespace emel::graph::assembler::action {

struct context {
  allocator::sm allocator_actor = {};
  const void * reserved_topology = nullptr;
  uint32_t reserved_node_count = 0;
  uint32_t reserved_tensor_count = 0;
  uint64_t reserved_required_buffer_bytes = 0;
  uint32_t topology_version = 0;
  uint8_t has_reserved_topology = 0;
};

}  // namespace emel::graph::assembler::action
