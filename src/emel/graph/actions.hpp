#pragma once

#include "emel/graph/context.hpp"

namespace emel::graph::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::graph::action
