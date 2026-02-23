#pragma once

#include "emel/tensor/view/context.hpp"

namespace emel::tensor::view::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::tensor::view::action
