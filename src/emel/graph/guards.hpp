#pragma once

#include "emel/graph/context.hpp"

namespace emel::graph::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::graph::guard
