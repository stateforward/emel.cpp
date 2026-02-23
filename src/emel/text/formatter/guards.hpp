#pragma once

#include "emel/text/formatter/context.hpp"

namespace emel::text::formatter::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::text::formatter::guard
