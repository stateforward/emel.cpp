#pragma once

#include "emel/kernel/events/context.hpp"

namespace emel::kernel::events::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::events::action
