#pragma once

#include "emel/kernel/vulkan/context.hpp"

namespace emel::kernel::vulkan::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::vulkan::action
