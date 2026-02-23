#pragma once

#include "emel/kernel/x86_64/context.hpp"

namespace emel::kernel::x86_64::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::x86_64::guard
