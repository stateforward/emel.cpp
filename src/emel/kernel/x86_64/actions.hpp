#pragma once

#include "emel/kernel/x86_64/context.hpp"

namespace emel::kernel::x86_64::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::x86_64::action
