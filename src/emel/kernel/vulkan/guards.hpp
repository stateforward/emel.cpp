#pragma once

#include "emel/kernel/vulkan/context.hpp"

namespace emel::kernel::vulkan::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::vulkan::guard
