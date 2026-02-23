#pragma once

#include "emel/graph/allocator/context.hpp"

namespace emel::graph::allocator::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::graph::allocator::action
