#pragma once

#include "emel/kernel/aarch64/context.hpp"

namespace emel::kernel::aarch64::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::aarch64::guard
