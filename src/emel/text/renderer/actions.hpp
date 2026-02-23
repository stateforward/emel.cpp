#pragma once

#include "emel/text/renderer/context.hpp"

namespace emel::text::renderer::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::text::renderer::action
