#pragma once

#include "emel/graph/graph/context.hpp"

namespace emel::graph::graph::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::graph::graph::guard
