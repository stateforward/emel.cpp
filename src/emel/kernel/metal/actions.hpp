#pragma once

#include "emel/kernel/metal/context.hpp"

namespace emel::kernel::metal::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::metal::action
