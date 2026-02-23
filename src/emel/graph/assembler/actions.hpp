#pragma once

#include "emel/graph/assembler/context.hpp"

namespace emel::graph::assembler::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::graph::assembler::action
