#pragma once

#include "emel/kernel/dispatch/context.hpp"

namespace emel::kernel::dispatch::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::dispatch::action
