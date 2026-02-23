#pragma once

#include "emel/kernel/any/context.hpp"

namespace emel::kernel::any::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::any::guard
