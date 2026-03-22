#pragma once

#include <cstdint>

#include "emel/graph/assembler/sm.hpp"
#include "emel/graph/processor/sm.hpp"
#include "emel/tensor/sm.hpp"

namespace emel::graph::action {

struct context {
  assembler::sm assembler_actor = {};
  processor::sm processor_actor = {};
  tensor::sm tensor_actor = {};
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::graph::action
