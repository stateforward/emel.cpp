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

inline void emit_token(const event::next & ev, const event::token & token, const uint32_t end) noexcept {
  ev.on_done(events::next_done{
      .token = token,
      .has_token = true,
      .next_cursor = next_cursor(ev.cursor, end),
  });
}

inline void emit_range_token(const event::next & ev,
                             const uint32_t start,
                             const uint32_t end,
                             const event::token_kind kind) noexcept {
  emit_token(ev, make_token(ev.cursor.input, start, end, kind), end);
}

inline uint32_t scan_quoted(const std::string_view input,
                            uint32_t pos,
                            const char terminator) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const uint32_t scan_start = static_cast<uint32_t>(pos + 1u);
  uint32_t end = scan_start;
  uint32_t active = static_cast<uint32_t>(scan_start <= size);
  uint32_t skip_escaped = 0u;
  for (uint32_t scan = scan_start; scan < size; ++scan) {
    const char c = input[scan];
    const uint32_t consume_char = active & (1u - skip_escaped);
    const uint32_t escaped =
        consume_char & static_cast<uint32_t>(c == '\\' && scan + 1u < size);
    const uint32_t matched =
        consume_char & static_cast<uint32_t>(c == terminator) & (1u - escaped);
    end += consume_char;
    end += escaped;
    active &= (1u - matched);
    skip_escaped = escaped;
  }
  return end;
}

inline uint32_t scan_braced_quantifier(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const uint32_t scan_start = static_cast<uint32_t>(pos + 1u);
  uint32_t end = scan_start;
  uint32_t active = static_cast<uint32_t>(scan_start <= size);
  for (uint32_t scan = scan_start; scan < size; ++scan) {
    const uint32_t consume_char = active;
    const uint32_t matched = consume_char & static_cast<uint32_t>(input[scan] == '}');
    end += consume_char;
    active &= (1u - matched);
  }
  return end;
}

}  // namespace detail

struct emit_layout_exhausted_unknown {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    detail::emit_token(ev, event::token{}, start);
  }
};

template <uint32_t width>
struct emit_newline_token_width {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = static_cast<uint32_t>(start + width);
    detail::emit_range_token(ev, start, end, event::token_kind::newline);
  }
};

struct emit_definition_operator {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    detail::emit_range_token(ev, start, static_cast<uint32_t>(start + 3u),
                             event::token_kind::definition_operator);
  }
};

template <event::token_kind kind>
struct emit_single_char_token {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
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
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = detail::scan_quoted(ev.cursor.input, start, '"');
    detail::emit_range_token(ev, start, end, event::token_kind::string_literal);
  }
};

struct emit_character_class {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = detail::scan_quoted(ev.cursor.input, start, ']');
    detail::emit_range_token(ev, start, end, event::token_kind::character_class);
  }
};

struct emit_braced_quantifier {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = detail::scan_braced_quantifier(ev.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::quantifier);
  }
};

struct emit_rule_reference_plain {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::rule_reference);
  }
};

struct emit_rule_reference_negated {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.cursor.input, start + 1u);
    detail::emit_range_token(ev, start, end, event::token_kind::rule_reference);
  }
};

struct emit_identifier {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t size = static_cast<uint32_t>(ev.cursor.input.size());
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    uint32_t end = static_cast<uint32_t>(start + 1u);
    uint32_t active = static_cast<uint32_t>(end <= size);
    for (uint32_t scan = end; scan < size; ++scan) {
      const uint32_t is_word =
          static_cast<uint32_t>(lexer::detail::is_word_char(ev.cursor.input[scan]));
      const uint32_t advance = active & is_word;
      end += advance;
      active = advance;
    }
    detail::emit_range_token(ev, start, end, event::token_kind::identifier);
  }
};

struct emit_eof {
  void operator()(const event::next & ev, context &) const noexcept {
    ev.on_done(events::next_done{
        .token = {},
        .has_token = false,
        .next_cursor = ev.cursor,
    });
  }
};

struct reject_invalid_next {
  void operator()(const event::next & ev, context &) const noexcept {
    ev.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct reject_invalid_cursor {
  void operator()(const event::next & ev, context &) const noexcept {
    ev.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct dispatch_unexpected_error {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.on_error; }) {
      (void)ev.on_error(events::next_error{error_code(error::internal_error)});
    }
  }
};

struct ignore_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

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
