#pragma once

#include "emel/text/jinja/parser/lexer/context.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"

namespace emel::text::jinja::parser::lexer::action {

struct emit_scanned_token {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_done(::emel::text::jinja::lexer::events::next_done{
        ev.request,
        ev.scan.token_value,
        true,
        ev.scan.next_cursor,
    });
  }
};

struct emit_scan_error {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_error(::emel::text::jinja::lexer::events::next_error{
        ev.request,
        ev.scan.err,
        ev.scan.error_pos,
    });
  }
};

struct emit_eof {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_done(::emel::text::jinja::lexer::events::next_done{
        ev.request,
        {},
        false,
        ev.request.cursor,
    });
  }
};

struct reject_invalid_next {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_error(::emel::text::jinja::lexer::events::next_error{
        ev.request,
        detail::error_code(error::invalid_request),
        ev.request.cursor.offset,
    });
  }
};

struct reject_invalid_cursor {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_error(::emel::text::jinja::lexer::events::next_error{
        ev.request,
        detail::error_code(error::invalid_request),
        ev.request.cursor.offset,
    });
  }
};

struct on_unexpected {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    ev.request.dispatch_error(::emel::text::jinja::lexer::events::next_error{
        ev.request,
        detail::error_code(error::internal_error),
        ev.request.cursor.offset,
    });
  }

  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr emit_scanned_token emit_scanned_token{};
inline constexpr emit_scan_error emit_scan_error{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::jinja::parser::lexer::action
