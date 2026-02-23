#pragma once

#include "emel/text/detokenizer/context.hpp"

namespace emel::text::detokenizer::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::text::detokenizer::guard
