#pragma once

#include "emel/tensor/view/context.hpp"

namespace emel::tensor::view::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::tensor::view::guard
