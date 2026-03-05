#pragma once

#include <cstddef>
#include <string_view>

#include "emel/text/jinja/lexer/detail.hpp"

namespace emel::text::jinja::parser::lexer::event {

struct next_ctx {
  ::emel::text::jinja::lexer::detail::scan_outcome scan = {};
  std::string_view source = {};
  size_t size = 0;
  size_t pos = 0;
  size_t text_start = 0;
  size_t text_end = 0;
  size_t text_trim_probe = 0;
  size_t string_start = 0;
  char string_terminal = '\0';
  bool handled = false;
};

struct next_runtime {
  const ::emel::text::jinja::lexer::event::next &request;
  next_ctx &ctx;
};

} // namespace emel::text::jinja::parser::lexer::event

namespace emel::text::jinja::parser::lexer::detail {

using ::emel::text::jinja::lexer::detail::error_code;
using ::emel::text::jinja::lexer::detail::normalize_source;
using ::emel::text::jinja::lexer::detail::scan_outcome;

} // namespace emel::text::jinja::parser::lexer::detail
