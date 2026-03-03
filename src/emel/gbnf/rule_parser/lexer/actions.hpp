#pragma once

#include <array>
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

inline bool noop_error_callback(const events::next_error &) noexcept {
  return true;
}

}  // namespace detail

struct emit_layout_exhausted_unknown {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    detail::emit_token(ev, event::token{}, start);
  }
};

struct emit_newline_token {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t size = static_cast<uint32_t>(ev.cursor.input.size());
    const bool crlf = ev.cursor.input[start] == '\r' && start + 1u < size &&
                      ev.cursor.input[start + 1u] == '\n';
    const std::array<uint32_t, 2> newline_steps = {1u, 2u};
    const uint32_t end = static_cast<uint32_t>(start + newline_steps[static_cast<size_t>(crlf)]);
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
    const uint32_t end = lexer::detail::scan_quoted(ev.cursor.input, start, '"');
    detail::emit_range_token(ev, start, end, event::token_kind::string_literal);
  }
};

struct emit_character_class {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = lexer::detail::scan_quoted(ev.cursor.input, start, ']');
    detail::emit_range_token(ev, start, end, event::token_kind::character_class);
  }
};

struct emit_braced_quantifier {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = lexer::detail::scan_braced_quantifier(ev.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::quantifier);
  }
};

struct emit_rule_reference {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    const uint32_t end = lexer::detail::scan_token_ref(ev.cursor.input, start);
    detail::emit_range_token(ev, start, end, event::token_kind::rule_reference);
  }
};

struct emit_identifier {
  void operator()(const event::next & ev, context &) const noexcept {
    const uint32_t size = static_cast<uint32_t>(ev.cursor.input.size());
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    uint32_t end = static_cast<uint32_t>(start + 1u);
    while (end < size && lexer::detail::is_word_char(ev.cursor.input[end])) {
      ++end;
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

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.on_error; }) {
      const size_t has_callback = static_cast<size_t>(static_cast<bool>(ev.on_error));
      const callback<bool(const events::next_error &)> callbacks[2] = {
          callback<bool(const events::next_error &)>::from<detail::noop_error_callback>(),
          ev.on_error,
      };
      (void)callbacks[has_callback](events::next_error{error_code(error::internal_error)});
    }
  }
};

inline constexpr emit_layout_exhausted_unknown emit_layout_exhausted_unknown{};
inline constexpr emit_newline_token emit_newline_token{};
inline constexpr emit_definition_operator emit_definition_operator{};
inline constexpr emit_alternation emit_alternation{};
inline constexpr emit_dot emit_dot{};
inline constexpr emit_open_group emit_open_group{};
inline constexpr emit_close_group emit_close_group{};
inline constexpr emit_quantifier emit_quantifier{};
inline constexpr emit_string_literal emit_string_literal{};
inline constexpr emit_character_class emit_character_class{};
inline constexpr emit_braced_quantifier emit_braced_quantifier{};
inline constexpr emit_rule_reference emit_rule_reference{};
inline constexpr emit_identifier emit_identifier{};
inline constexpr emit_unknown emit_unknown{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::lexer::action
