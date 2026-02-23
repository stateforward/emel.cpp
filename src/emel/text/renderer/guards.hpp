#pragma once

#include "emel/text/renderer/context.hpp"

namespace emel::text::renderer::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::text::renderer::guard
