#pragma once

#include <span>

#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader::action {

struct context {
  geometry probed = {};
  std::span<const uint8_t> probed_file_image = {};
  std::span<tensor_view> tensors = {};
};

} // namespace emel::cact::loader::action
