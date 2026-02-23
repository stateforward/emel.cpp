#pragma once

#include "emel/kernel/ops/context.hpp"

namespace emel::kernel::ops::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::ops::guard
