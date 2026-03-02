#pragma once

#include "emel/text/jinja/parser/lexer/context.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"
#include "emel/text/jinja/parser/lexer/errors.hpp"

namespace emel::text::jinja::parser::lexer::guard {

struct invalid_next {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.cursor.source.data() == nullptr ||
           !static_cast<bool>(ev.request.dispatch_done) ||
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_cursor_position {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.cursor.offset > ev.request.cursor.source.size();
  }
};

struct scan_failed {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.scan.err != detail::error_code(error::none);
  }
};

struct scan_has_token {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.scan.err == detail::error_code(error::none) && ev.scan.has_token;
  }
};

struct scan_at_eof {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.scan.err == detail::error_code(error::none) && !ev.scan.has_token;
  }
};

} // namespace emel::text::jinja::parser::lexer::guard
