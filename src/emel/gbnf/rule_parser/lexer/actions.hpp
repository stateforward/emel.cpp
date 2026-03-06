#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gbnf/rule_parser/lexer/context.hpp"
#include "emel/gbnf/rule_parser/lexer/detail.hpp"
#include "emel/gbnf/rule_parser/lexer/errors.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"

namespace emel::gbnf::rule_parser::lexer::action {

inline constexpr int32_t error_code(const emel::gbnf::rule_parser::lexer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

namespace detail {

inline event::token make_token(const std::string_view input,
                               const uint32_t start,
                               const uint32_t end,
                               const event::token_kind kind) noexcept {
  return event::token{
      .kind = kind,
      .text = input.substr(start, end - start),
      .start = start,
      .end = end,
  };
}

inline lexer::cursor next_cursor(const lexer::cursor & cursor, const uint32_t next_offset) noexcept {
  lexer::cursor advanced = cursor;
  advanced.offset = next_offset;
  advanced.token_count += 1;
  return advanced;
}

inline void emit_token(const event::scan_next & ev, const event::token & token, const uint32_t end) noexcept {
  ev.request.on_done(events::next_done{
      .token = token,
      .has_token = true,
      .next_cursor = next_cursor(ev.request.cursor, end),
  });
}

inline void emit_range_token(const event::scan_next & ev,
                             const uint32_t start,
                             const uint32_t end,
                             const event::token_kind kind) noexcept {
  emit_token(ev, make_token(ev.request.cursor.input, start, end, kind), end);
}

inline uint32_t scan_quoted(const std::string_view input,
                            uint32_t pos,
                            const char terminator) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  uint32_t scan = static_cast<uint32_t>(pos + 1u);
  while (scan < size) {
    const char c = input[scan];
    ++scan;
    if (c == '\\' && scan < size) {
      ++scan;
      continue;
    }
    if (c == terminator) {
      break;
    }
  }
  return scan;
}

inline uint32_t scan_braced_quantifier(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  uint32_t scan = static_cast<uint32_t>(pos + 1u);
  while (scan < size) {
    if (input[scan] == '}') {
      ++scan;
      break;
    }
    ++scan;
  }
  return scan;
}

}  // namespace detail

struct prepare_scan {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    ev.ctx.start = lexer::detail::token_start(ev.request.cursor);
    ev.ctx.has_input = ev.ctx.start < ev.request.cursor.input.size();
    ev.ctx.first_char = ev.ctx.has_input ? ev.request.cursor.input[ev.ctx.start] : '\0';
  }
};

struct emit_layout_exhausted_unknown {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    detail::emit_token(ev, event::token{}, ev.ctx.start);
  }
};

template <uint32_t width>
struct emit_newline_token_width {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = static_cast<uint32_t>(start + width);
    detail::emit_range_token(ev, start, end, event::token_kind::newline);
  }
};

struct emit_definition_operator {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    detail::emit_range_token(ev, start, static_cast<uint32_t>(start + 3u),
                             event::token_kind::definition_operator);
  }
};

template <event::token_kind kind>
struct emit_single_char_token {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    detail::emit_range_token(ev, start, static_cast<uint32_t>(start + 1u), kind);
  }
};

struct emit_alternation : emit_single_char_token<event::token_kind::alternation> {};
struct emit_dot : emit_single_char_token<event::token_kind::dot> {};
struct emit_open_group : emit_single_char_token<event::token_kind::open_group> {};
struct emit_close_group : emit_single_char_token<event::token_kind::close_group> {};
struct emit_quantifier : emit_single_char_token<event::token_kind::quantifier> {};
struct emit_unknown : emit_single_char_token<event::token_kind::unknown> {};

struct emit_string_literal {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = detail::scan_quoted(ev.request.cursor.input, start, '"');
    detail::emit_range_token(ev, start, end, event::token_kind::string_literal);
  }
};

struct emit_character_class {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = detail::scan_quoted(ev.request.cursor.input, start, ']');
    detail::emit_range_token(ev, start, end, event::token_kind::character_class);
  }
};

struct emit_braced_quantifier {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = detail::scan_braced_quantifier(ev.request.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::quantifier);
  }
};

struct emit_rule_reference_plain {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.request.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::rule_reference);
  }
};

struct emit_rule_reference_negated {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.request.cursor.input, start + 1u);
    detail::emit_range_token(ev, start, end, event::token_kind::rule_reference);
  }
};

struct emit_identifier {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    const uint32_t start = ev.ctx.start;
    const uint32_t size = static_cast<uint32_t>(ev.request.cursor.input.size());
    uint32_t end = static_cast<uint32_t>(start + 1u);
    while (end < size && lexer::detail::is_word_char(ev.request.cursor.input[end])) {
      ++end;
    }
    detail::emit_range_token(ev, start, end, event::token_kind::identifier);
  }
};

struct emit_eof {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    ev.request.on_done(events::next_done{
        .token = {},
        .has_token = false,
        .next_cursor = ev.request.cursor,
    });
  }
};

struct reject_invalid_next {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    ev.request.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct reject_invalid_cursor {
  void operator()(const event::scan_next & ev, context &) const noexcept {
    ev.request.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct dispatch_unexpected_error {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.request.on_error; }) {
      (void)ev.request.on_error(events::next_error{error_code(error::internal_error)});
    }
  }
};

struct ignore_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr prepare_scan prepare_scan{};
inline constexpr emit_layout_exhausted_unknown emit_layout_exhausted_unknown{};
inline constexpr emit_newline_token_width<1u> emit_newline_single_token{};
inline constexpr emit_newline_token_width<2u> emit_newline_crlf_token{};
inline constexpr emit_definition_operator emit_definition_operator{};
inline constexpr emit_alternation emit_alternation{};
inline constexpr emit_dot emit_dot{};
inline constexpr emit_open_group emit_open_group{};
inline constexpr emit_close_group emit_close_group{};
inline constexpr emit_quantifier emit_quantifier{};
inline constexpr emit_string_literal emit_string_literal{};
inline constexpr emit_character_class emit_character_class{};
inline constexpr emit_braced_quantifier emit_braced_quantifier{};
inline constexpr emit_rule_reference_plain emit_rule_reference_plain{};
inline constexpr emit_rule_reference_negated emit_rule_reference_negated{};
inline constexpr emit_identifier emit_identifier{};
inline constexpr emit_unknown emit_unknown{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr dispatch_unexpected_error dispatch_unexpected_error{};
inline constexpr ignore_unexpected ignore_unexpected{};

}  // namespace emel::gbnf::rule_parser::lexer::action
