#pragma once

#include "emel/graph/assembler/context.hpp"

namespace emel::graph::assembler::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::graph::assembler::guard
