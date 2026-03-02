#pragma once

#include "emel/text/jinja/parser/lexer/sm.hpp"

namespace emel::text::jinja::parser::action {

struct context {
  emel::text::jinja::parser::lexer::sm lexer = {};
};

} // namespace emel::text::jinja::parser::action
