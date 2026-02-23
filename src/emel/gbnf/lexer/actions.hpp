#pragma once

#include "emel/gbnf/lexer/context.hpp"

namespace emel::gbnf::lexer::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::gbnf::lexer::action
