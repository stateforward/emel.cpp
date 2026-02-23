#pragma once

#include "emel/graph/allocator/context.hpp"

namespace emel::graph::allocator::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::graph::allocator::guard
