#pragma once

#include "emel/kernel/context.hpp"

namespace emel::kernel::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::guard
