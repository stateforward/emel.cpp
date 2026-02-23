#pragma once

#include "emel/kernel/cuda/context.hpp"

namespace emel::kernel::cuda::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::cuda::guard
