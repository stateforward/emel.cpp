#pragma once

#include "emel/kernel/aarch64/context.hpp"

namespace emel::kernel::aarch64::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::aarch64::action
