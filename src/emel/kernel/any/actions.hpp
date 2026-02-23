#pragma once

#include "emel/kernel/any/context.hpp"

namespace emel::kernel::any::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::any::action
