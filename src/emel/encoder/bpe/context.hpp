#pragma once

#include <vector>

#include "emel/encoder/context.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::bpe::action {

struct context : emel::encoder::action::context {
  emel::model::data::TokenizerPre bpe_pre_id = emel::model::data::TokenizerPre::DEFAULT;
  std::vector<std::string> bpe_regex_exprs = {};
};

}  // namespace emel::encoder::bpe::action
