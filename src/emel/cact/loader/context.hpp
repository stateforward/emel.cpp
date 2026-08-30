#pragma once

#include <span>

#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader::action {

struct context {
  geometry probed = {};
  std::span<tensor_view> tensors = {};
};

} // namespace emel::cact::loader::action
