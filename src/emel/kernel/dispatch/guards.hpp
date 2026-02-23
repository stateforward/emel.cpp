#pragma once

#include "emel/kernel/dispatch/context.hpp"

namespace emel::kernel::dispatch::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::dispatch::guard
