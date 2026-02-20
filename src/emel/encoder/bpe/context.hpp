#pragma once

#include <string>
#include <vector>

#include "emel/encoder/context.hpp"
#include "emel/encoder/types.hpp"

namespace emel::encoder::bpe::action {

struct context : emel::encoder::action::context {
  std::string bpe_pre = {};
  std::vector<std::string> bpe_regex_exprs = {};
};

}  // namespace emel::encoder::bpe::action
