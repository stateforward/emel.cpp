#pragma once

#include "emel/text/detokenizer/context.hpp"

namespace emel::text::detokenizer::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::text::detokenizer::action
