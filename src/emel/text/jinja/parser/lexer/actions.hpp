#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "emel/text/jinja/parser/lexer/context.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"

namespace emel::text::jinja::parser::lexer::action {

namespace helper {

inline void reset_phase(event::next_runtime &ev) noexcept {
  ev.ctx.handled = false;
  ev.ctx.scan.token_value = {};
  ev.ctx.scan.has_token = false;
  ev.ctx.scan.err = detail::error_code(error::none);
  ev.ctx.scan.error_pos = 0;
  ev.ctx.scan.next_cursor = ev.request.cursor;
  ev.ctx.scan.next_cursor.offset = ev.ctx.pos;
}

inline void trim_text_before_opening_block(const std::string_view source,
                                           const size_t start,
                                           size_t &end) noexcept {
  size_t current = end;
  bool keep_trimming = true;
  while (current > start && keep_trimming) {
    const char c = source[current - 1];
    const bool at_start = current == 1u;
    const bool newline = c == '\n';

    if (at_start) {
      end = 0u;
      keep_trimming = false;
    } else if (newline) {
      end = current;
      keep_trimming = false;
    } else if (::emel::text::jinja::lexer::detail::is_space(c)) {
      --current;
    } else {
      keep_trimming = false;
    }
  }
}

inline void emit_scanned_token(const event::next_runtime &ev) noexcept {
  ev.request.dispatch_done(::emel::text::jinja::lexer::events::next_done{
      ev.request,
      ev.ctx.scan.token_value,
      true,
      ev.ctx.scan.next_cursor,
  });
}

inline void emit_scan_error(const event::next_runtime &ev) noexcept {
  ev.request.dispatch_error(::emel::text::jinja::lexer::events::next_error{
      ev.request,
      ev.ctx.scan.err,
      ev.ctx.scan.error_pos,
  });
}

inline void emit_eof(const event::next_runtime &ev) noexcept {
  ev.request.dispatch_done(::emel::text::jinja::lexer::events::next_done{
      ev.request,
      {},
      false,
      ev.ctx.scan.next_cursor,
  });
}

} // namespace helper

struct begin_scan {
  void operator()(event::next_runtime ev, context &) const noexcept {
    ev.ctx.source = ev.request.cursor.source;
    ev.ctx.size = ev.request.cursor.source.size();
    ev.ctx.pos = ev.request.cursor.offset;
    helper::reset_phase(ev);
  }
};

struct scan_text_boundary {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const std::string_view source = ev.ctx.source;
    const size_t size = ev.ctx.size;
    size_t &pos = ev.ctx.pos;

    const size_t start = pos;
    size_t end = start;
    while (pos < size &&
           !(source[pos] == '{' &&
             ::emel::text::jinja::lexer::detail::next_pos_is(source, pos, {'%', '{', '#'}))) {
      end = ++pos;
    }

    ev.ctx.handled = true;
    ev.ctx.text_start = start;
    ev.ctx.text_end = end;
  }
};

struct trim_text_before_opening_block {
  void operator()(event::next_runtime ev, context &) const noexcept {
    helper::trim_text_before_opening_block(ev.ctx.source, ev.ctx.text_start, ev.ctx.text_end);
  }
};

struct materialize_text_token {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.scan.token_value = token{
        token_type::text,
        std::string(ev.ctx.source.substr(ev.ctx.text_start, ev.ctx.text_end - ev.ctx.text_start)),
        ev.ctx.text_start,
    };
  }
};

struct trim_text_leading_newline {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.scan.token_value.value.erase(ev.ctx.scan.token_value.value.begin());
  }
};

struct lstrip_text_token {
  void operator()(event::next_runtime ev, context &) const {
    ::emel::text::jinja::lexer::detail::string_lstrip(ev.ctx.scan.token_value.value, " \t\r\n");
  }
};

struct rstrip_text_token {
  void operator()(event::next_runtime ev, context &) const {
    ::emel::text::jinja::lexer::detail::string_rstrip(ev.ctx.scan.token_value.value, " \t\r\n");
  }
};

struct lstrip_and_rstrip_text_token {
  void operator()(event::next_runtime ev, context &) const {
    ::emel::text::jinja::lexer::detail::string_lstrip(ev.ctx.scan.token_value.value, " \t\r\n");
    ::emel::text::jinja::lexer::detail::string_rstrip(ev.ctx.scan.token_value.value, " \t\r\n");
  }
};

struct finalize_text_boundary_token {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.scan.has_token = !ev.ctx.scan.token_value.value.empty();
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
  }
};

struct scan_comment {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const std::string_view source = ev.ctx.source;
    const size_t size = ev.ctx.size;
    size_t &pos = ev.ctx.pos;

    const size_t start = pos;
    pos += 2u;
    std::string comment;
    while (pos < size &&
           !(source[pos] == '#' &&
             ::emel::text::jinja::lexer::detail::next_pos_is(source, pos, {'}'}))) {
      comment.push_back(source[pos]);
      ++pos;
    }

    ev.ctx.handled = true;
    ev.ctx.scan.token_value = token{token_type::comment, std::move(comment), start};
  }
};

struct finalize_comment_token {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.pos += 2u;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
  }
};

struct mark_comment_unterminated {
  void operator()(event::next_runtime ev, context &) const noexcept {
    ::emel::text::jinja::lexer::detail::set_error(ev.ctx.scan, ev.ctx.pos);
  }
};

struct scan_trim_prefix {
  void operator()(event::next_runtime ev, context &) const noexcept {
    helper::reset_phase(ev);
    ++ev.ctx.pos;
  }
};

struct scan_spaces {
  void operator()(event::next_runtime ev, context &) const noexcept {
    helper::reset_phase(ev);

    while (ev.ctx.pos < ev.ctx.size &&
           ::emel::text::jinja::lexer::detail::is_space(ev.ctx.source[ev.ctx.pos])) {
      ++ev.ctx.pos;
    }
  }
};

struct mark_no_token_eof {
  void operator()(event::next_runtime ev, context &) const noexcept {
    ev.ctx.handled = true;
    ::emel::text::jinja::lexer::detail::emit_no_token_cursor(
        ev.ctx.scan,
        ev.request.cursor,
        ev.ctx.pos);
  }
};

struct scan_unary {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const std::string_view source = ev.ctx.source;
    size_t &pos = ev.ctx.pos;
    const char ch = source[pos];
    const size_t start = pos;
    ++pos;
    std::string num = ::emel::text::jinja::lexer::detail::consume_numeric(source, pos);
    std::string value;
    value.reserve(num.size() + 1u);
    value.push_back(ch);
    value += num;

    ev.ctx.handled = true;
    ev.ctx.scan.token_value = token{token_type::unary_operator, std::move(value), start};
  }
};

struct emit_unary_numeric_token {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value.type = token_type::numeric_literal;
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
    helper::emit_scanned_token(ev);
  }
};

struct emit_unary_operator_token {
  void operator()(event::next_runtime ev, context &) const {
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value.type = token_type::unary_operator;
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
    helper::emit_scanned_token(ev);
  }
};

template <token_type mapped_token, char... seq_chars>
struct scan_fixed_mapping {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    constexpr std::array<char, sizeof...(seq_chars)> seq = {seq_chars...};
    const std::string_view token_text(seq.data(), seq.size());
    const size_t pos = ev.ctx.pos;

    ev.ctx.handled = true;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value = token{mapped_token, std::string(token_text), pos};
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        pos + token_text.size(),
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
    ev.ctx.pos = pos + token_text.size();
  }
};

struct scan_mapping_close_curly {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const size_t pos = ev.ctx.pos;
    ev.ctx.handled = true;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value = token{token_type::close_curly_bracket, "}", pos};
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        pos + 1u,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
    ev.ctx.pos = pos + 1u;
  }
};

struct scan_string {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const std::string_view source = ev.ctx.source;
    size_t &pos = ev.ctx.pos;

    const size_t start = pos;
    const char terminal = source[pos];
    ++pos;
    std::string value =
        ::emel::text::jinja::lexer::detail::consume_escaped_until(source, pos, terminal,
                                                                   ev.ctx.scan);
    ev.ctx.handled = true;
    ev.ctx.scan.token_value = token{token_type::string_literal, std::move(value), start};
  }
};

struct finalize_string_token {
  void operator()(event::next_runtime ev, context &) const {
    ++ev.ctx.pos;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
  }
};

struct mark_string_unterminated {
  void operator()(event::next_runtime ev, context &) const noexcept {
    ::emel::text::jinja::lexer::detail::set_error(ev.ctx.scan, ev.ctx.pos);
  }
};

struct scan_numeric {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const size_t start = ev.ctx.pos;
    std::string value =
        ::emel::text::jinja::lexer::detail::consume_numeric(ev.ctx.source, ev.ctx.pos);

    ev.ctx.handled = true;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value = token{token_type::numeric_literal, std::move(value), start};
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
  }
};

struct scan_word {
  void operator()(event::next_runtime ev, context &) const {
    helper::reset_phase(ev);

    const size_t start = ev.ctx.pos;
    std::string value;
    while (ev.ctx.pos < ev.ctx.size &&
           ::emel::text::jinja::lexer::detail::is_word(ev.ctx.source[ev.ctx.pos])) {
      value.push_back(ev.ctx.source[ev.ctx.pos]);
      ++ev.ctx.pos;
    }

    ev.ctx.handled = true;
    ev.ctx.scan.has_token = true;
    ev.ctx.scan.token_value = token{token_type::identifier, std::move(value), start};
    ev.ctx.scan.next_cursor = ::emel::text::jinja::lexer::detail::emit_cursor(
        ev.request.cursor,
        ev.ctx.pos,
        ev.ctx.scan.token_value.type,
        ev.ctx.scan.token_value.value);
  }
};

struct mark_invalid_character {
  void operator()(event::next_runtime ev, context &) const noexcept {
    helper::reset_phase(ev);
    ev.ctx.handled = true;
    ::emel::text::jinja::lexer::detail::set_error(ev.ctx.scan, ev.ctx.pos);
  }
};

struct emit_scanned_token {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    helper::emit_scanned_token(ev);
  }
};

struct emit_scan_error {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    helper::emit_scan_error(ev);
  }
};

struct emit_eof {
  void operator()(const event::next_runtime &ev, context &) const noexcept {
    helper::emit_eof(ev);
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

inline constexpr begin_scan begin_scan{};
inline constexpr scan_text_boundary scan_text_boundary{};
inline constexpr trim_text_before_opening_block trim_text_before_opening_block{};
inline constexpr materialize_text_token materialize_text_token{};
inline constexpr trim_text_leading_newline trim_text_leading_newline{};
inline constexpr lstrip_text_token lstrip_text_token{};
inline constexpr rstrip_text_token rstrip_text_token{};
inline constexpr lstrip_and_rstrip_text_token lstrip_and_rstrip_text_token{};
inline constexpr finalize_text_boundary_token finalize_text_boundary_token{};
inline constexpr scan_comment scan_comment{};
inline constexpr finalize_comment_token finalize_comment_token{};
inline constexpr mark_comment_unterminated mark_comment_unterminated{};
inline constexpr scan_trim_prefix scan_trim_prefix{};
inline constexpr scan_spaces scan_spaces{};
inline constexpr mark_no_token_eof mark_no_token_eof{};
inline constexpr scan_unary scan_unary{};
inline constexpr emit_unary_numeric_token emit_unary_numeric_token{};
inline constexpr emit_unary_operator_token emit_unary_operator_token{};
inline constexpr scan_fixed_mapping<token_type::open_statement, '{', '%', '-'>
    scan_mapping_open_statement_trim{};
inline constexpr scan_fixed_mapping<token_type::close_statement, '-', '%', '}'>
    scan_mapping_close_statement_trim{};
inline constexpr scan_fixed_mapping<token_type::open_expression, '{', '{', '-'>
    scan_mapping_open_expression_trim{};
inline constexpr scan_fixed_mapping<token_type::close_expression, '-', '}', '}'>
    scan_mapping_close_expression_trim{};
inline constexpr scan_fixed_mapping<token_type::open_statement, '{', '%'>
    scan_mapping_open_statement{};
inline constexpr scan_fixed_mapping<token_type::close_statement, '%', '}'>
    scan_mapping_close_statement{};
inline constexpr scan_fixed_mapping<token_type::open_expression, '{', '{'>
    scan_mapping_open_expression{};
inline constexpr scan_fixed_mapping<token_type::close_expression, '}', '}'>
    scan_mapping_close_expression{};
inline constexpr scan_fixed_mapping<token_type::open_paren, '('> scan_mapping_open_paren{};
inline constexpr scan_fixed_mapping<token_type::close_paren, ')'> scan_mapping_close_paren{};
inline constexpr scan_fixed_mapping<token_type::open_curly_bracket, '{'>
    scan_mapping_open_curly_bracket{};
inline constexpr scan_fixed_mapping<token_type::close_curly_bracket, '}'>
    scan_mapping_close_curly_bracket{};
inline constexpr scan_fixed_mapping<token_type::open_square_bracket, '['>
    scan_mapping_open_square_bracket{};
inline constexpr scan_fixed_mapping<token_type::close_square_bracket, ']'>
    scan_mapping_close_square_bracket{};
inline constexpr scan_fixed_mapping<token_type::comma, ','> scan_mapping_comma{};
inline constexpr scan_fixed_mapping<token_type::dot, '.'> scan_mapping_dot{};
inline constexpr scan_fixed_mapping<token_type::colon, ':'> scan_mapping_colon{};
inline constexpr scan_fixed_mapping<token_type::pipe, '|'> scan_mapping_pipe{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '<', '='>
    scan_mapping_less_equal{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '>', '='>
    scan_mapping_greater_equal{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '=', '='>
    scan_mapping_equal_equal{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '!', '='>
    scan_mapping_bang_equal{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '<'>
    scan_mapping_less{};
inline constexpr scan_fixed_mapping<token_type::comparison_binary_operator, '>'>
    scan_mapping_greater{};
inline constexpr scan_fixed_mapping<token_type::additive_binary_operator, '+'>
    scan_mapping_plus{};
inline constexpr scan_fixed_mapping<token_type::additive_binary_operator, '-'>
    scan_mapping_minus{};
inline constexpr scan_fixed_mapping<token_type::additive_binary_operator, '~'>
    scan_mapping_tilde{};
inline constexpr scan_fixed_mapping<token_type::multiplicative_binary_operator, '*'>
    scan_mapping_star{};
inline constexpr scan_fixed_mapping<token_type::multiplicative_binary_operator, '/'>
    scan_mapping_slash{};
inline constexpr scan_fixed_mapping<token_type::multiplicative_binary_operator, '%'>
    scan_mapping_percent{};
inline constexpr scan_fixed_mapping<token_type::equals, '='> scan_mapping_equals{};
inline constexpr scan_mapping_close_curly scan_mapping_close_curly{};
inline constexpr scan_string scan_string{};
inline constexpr finalize_string_token finalize_string_token{};
inline constexpr mark_string_unterminated mark_string_unterminated{};
inline constexpr scan_numeric scan_numeric{};
inline constexpr scan_word scan_word{};
inline constexpr mark_invalid_character mark_invalid_character{};
inline constexpr emit_scanned_token emit_scanned_token{};
inline constexpr emit_scan_error emit_scan_error{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::jinja::parser::lexer::action
