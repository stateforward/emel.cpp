#pragma once

#include "emel/kernel/cuda/context.hpp"

namespace emel::kernel::cuda::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::cuda::action
