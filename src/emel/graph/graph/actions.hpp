#pragma once

#include "emel/graph/graph/context.hpp"

namespace emel::graph::graph::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::graph::graph::action
