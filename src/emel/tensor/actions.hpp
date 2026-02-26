#pragma once

#include "emel/tensor/tensor/context.hpp"

namespace emel::tensor::tensor::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::tensor::tensor::action
