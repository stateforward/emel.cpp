#pragma once

#include "emel/kernel/context.hpp"

namespace emel::kernel::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::action
