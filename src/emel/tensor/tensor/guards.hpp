#pragma once

#include "emel/tensor/tensor/context.hpp"

namespace emel::tensor::tensor::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::tensor::tensor::guard
