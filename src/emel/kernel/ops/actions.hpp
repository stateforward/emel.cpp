#pragma once

#include "emel/kernel/ops/context.hpp"

namespace emel::kernel::ops::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::ops::action
