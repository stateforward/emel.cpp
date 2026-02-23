#pragma once

#include "emel/kernel/events/context.hpp"

namespace emel::kernel::events::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::events::guard
