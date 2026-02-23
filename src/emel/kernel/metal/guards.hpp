#pragma once

#include "emel/kernel/metal/context.hpp"

namespace emel::kernel::metal::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::metal::guard
