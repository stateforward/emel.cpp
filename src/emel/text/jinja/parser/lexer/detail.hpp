#pragma once

#include "emel/text/jinja/lexer/detail.hpp"

namespace emel::text::jinja::parser::lexer::event {

struct next_runtime {
  const ::emel::text::jinja::lexer::event::next &request;
  const ::emel::text::jinja::lexer::detail::scan_outcome &scan;
};

} // namespace emel::text::jinja::parser::lexer::event

namespace emel::text::jinja::parser::lexer::detail {

using ::emel::text::jinja::lexer::detail::error_code;
using ::emel::text::jinja::lexer::detail::normalize_source;
using ::emel::text::jinja::lexer::detail::scan_outcome;

} // namespace emel::text::jinja::parser::lexer::detail
