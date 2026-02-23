#pragma once

#include "emel/text/conditioner/context.hpp"

namespace emel::text::conditioner::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::text::conditioner::action
