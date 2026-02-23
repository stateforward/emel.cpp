#pragma once

#include "emel/text/conditioner/context.hpp"

namespace emel::text::conditioner::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::text::conditioner::guard
