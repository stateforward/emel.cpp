#pragma once

#include "emel/text/formatter/context.hpp"

namespace emel::text::formatter::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::text::formatter::action
