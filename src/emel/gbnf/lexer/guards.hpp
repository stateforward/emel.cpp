#pragma once

#include "emel/gbnf/lexer/context.hpp"

namespace emel::gbnf::lexer::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::gbnf::lexer::guard
