#pragma once

#include <array>
#include <string_view>

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

struct valid_next {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !invalid_next{}(ev, ctx);
  }
};

struct invalid_cursor_position {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return valid_next{}(ev, ctx) &&
           ev.request.cursor.offset > ev.request.cursor.source.size();
  }
};

struct valid_cursor_position {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return valid_next{}(ev, ctx) && !invalid_cursor_position{}(ev, ctx);
  }
};

struct phase_failed {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled && ev.ctx.scan.err != detail::error_code(error::none);
  }
};

struct phase_has_token {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled && ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.scan.has_token;
  }
};

struct phase_at_eof {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled && ev.ctx.scan.err == detail::error_code(error::none) &&
           !ev.ctx.scan.has_token;
  }
};

struct phase_unhandled {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return !ev.ctx.handled;
  }
};

struct at_text_boundary {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ::emel::text::jinja::lexer::detail::at_text_boundary(
        ev.request.cursor.last_token_type);
  }
};

struct not_at_text_boundary {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !at_text_boundary{}(ev, ctx);
  }
};

struct text_token_non_empty {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           !ev.ctx.scan.token_value.value.empty();
  }
};

struct text_token_empty {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_token_non_empty{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

struct text_can_trim_leading_newline {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.request.cursor.last_block_can_trim_newline &&
           !ev.ctx.scan.token_value.value.empty() &&
           ev.ctx.scan.token_value.value.front() == '\n';
  }
};

struct text_skip_trim_leading_newline {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_can_trim_leading_newline{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

struct text_opening_block_ahead {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.pos < ev.ctx.size &&
           ev.ctx.source[ev.ctx.pos] == '{' &&
           ::emel::text::jinja::lexer::detail::next_pos_is(ev.ctx.source, ev.ctx.pos, {'%', '#', '-'});
  }
};

struct text_opening_block_not_ahead {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_opening_block_ahead{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

struct text_last_block_rstrip_enabled {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.request.cursor.last_block_rstrip;
  }
};

struct text_last_block_rstrip_disabled {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_last_block_rstrip_enabled{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

struct text_next_block_lstrip_marker_present {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.pos < ev.ctx.size &&
           ev.ctx.source[ev.ctx.pos] == '{' &&
           ::emel::text::jinja::lexer::detail::next_pos_is(ev.ctx.source,
                                                           ev.ctx.pos,
                                                           {'{', '%', '#'}) &&
           ::emel::text::jinja::lexer::detail::next_pos_is(ev.ctx.source, ev.ctx.pos, {'-'}, 2u);
  }
};

struct text_next_block_lstrip_marker_absent {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_next_block_lstrip_marker_present{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

struct text_apply_lstrip_and_rstrip {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return text_last_block_rstrip_enabled{}(ev, ctx) &&
           text_next_block_lstrip_marker_present{}(ev, ctx);
  }
};

struct text_apply_lstrip_only {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return text_last_block_rstrip_enabled{}(ev, ctx) &&
           text_next_block_lstrip_marker_absent{}(ev, ctx);
  }
};

struct text_apply_rstrip_only {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return text_last_block_rstrip_disabled{}(ev, ctx) &&
           text_next_block_lstrip_marker_present{}(ev, ctx);
  }
};

struct text_apply_no_strip {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return text_last_block_rstrip_disabled{}(ev, ctx) &&
           text_next_block_lstrip_marker_absent{}(ev, ctx);
  }
};

struct starts_comment {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size &&
           ev.ctx.source[ev.ctx.pos] == '{' &&
           ::emel::text::jinja::lexer::detail::next_pos_is(ev.ctx.source, ev.ctx.pos, {'#'});
  }
};

struct not_starts_comment {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !starts_comment{}(ev, ctx);
  }
};

struct comment_terminated {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size &&
           ev.ctx.source[ev.ctx.pos] == '#' &&
           ::emel::text::jinja::lexer::detail::next_pos_is(ev.ctx.source, ev.ctx.pos, {'}'});
  }
};

struct comment_unterminated {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !comment_terminated{}(ev, ctx);
  }
};

struct starts_trim_prefix {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size &&
           ev.ctx.source[ev.ctx.pos] == '-' &&
           (ev.request.cursor.last_token_type == token_type::open_expression ||
            ev.request.cursor.last_token_type == token_type::open_statement);
  }
};

struct not_starts_trim_prefix {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !starts_trim_prefix{}(ev, ctx);
  }
};

struct cursor_at_end {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos >= ev.ctx.size;
  }
};

struct cursor_not_at_end {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !cursor_at_end{}(ev, ctx);
  }
};

struct unary_candidate {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    const bool in_range = ev.ctx.pos < ev.ctx.size;
    const char ch = ::emel::text::jinja::lexer::detail::view_char_at_or(
        ev.ctx.source, ev.ctx.pos, '\0');
    const bool is_sign = ch == '-' || ch == '+';
    const bool not_closing_block =
        !::emel::text::jinja::lexer::detail::is_closing_block(ev.ctx.source, ev.ctx.pos);
    return in_range && is_sign && not_closing_block;
  }
};

struct unary_not_candidate {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !unary_candidate{}(ev, ctx);
  }
};

struct unary_prefix_context_invalid {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.cursor.last_token_type == token_type::text ||
           ev.request.cursor.last_token_type == token_type::eof;
  }
};

struct unary_prefix_context_valid {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !unary_prefix_context_invalid{}(ev, ctx);
  }
};

struct unary_prefix_allowed {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ::emel::text::jinja::lexer::detail::unary_prefix_allowed(
        ev.request.cursor.last_token_type);
  }
};

struct unary_prefix_disallowed {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !unary_prefix_allowed{}(ev, ctx);
  }
};

struct unary_numeric_suffix_present {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.scan.token_value.value.size() > 1u;
  }
};

struct unary_numeric_suffix_absent {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !unary_numeric_suffix_present{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
  }
};

template <char... seq_chars>
struct mapping_sequence {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    constexpr std::array<char, sizeof...(seq_chars)> seq = {seq_chars...};
    const std::string_view token_text(seq.data(), seq.size());
    return ev.ctx.pos + token_text.size() <= ev.ctx.size &&
           ev.ctx.source.compare(ev.ctx.pos, token_text.size(), token_text) == 0;
  }
};

using mapping_open_statement_trim = mapping_sequence<'{', '%', '-'>;
using mapping_close_statement_trim = mapping_sequence<'-', '%', '}'>;
using mapping_open_expression_trim = mapping_sequence<'{', '{', '-'>;
using mapping_close_expression_trim = mapping_sequence<'-', '}', '}'>;
using mapping_open_statement = mapping_sequence<'{', '%'>;
using mapping_close_statement = mapping_sequence<'%', '}'>;
using mapping_open_expression = mapping_sequence<'{', '{'>;
using mapping_close_expression = mapping_sequence<'}', '}'>;
using mapping_open_paren = mapping_sequence<'('>;
using mapping_close_paren = mapping_sequence<')'>;
using mapping_open_curly_bracket = mapping_sequence<'{'>;
using mapping_close_curly_bracket = mapping_sequence<'}'>;
using mapping_open_square_bracket = mapping_sequence<'['>;
using mapping_close_square_bracket = mapping_sequence<']'>;
using mapping_comma = mapping_sequence<','>;
using mapping_dot = mapping_sequence<'.'>;
using mapping_colon = mapping_sequence<':'>;
using mapping_pipe = mapping_sequence<'|'>;
using mapping_less_equal = mapping_sequence<'<', '='>;
using mapping_greater_equal = mapping_sequence<'>', '='>;
using mapping_equal_equal = mapping_sequence<'=', '='>;
using mapping_bang_equal = mapping_sequence<'!', '='>;
using mapping_less = mapping_sequence<'<'>;
using mapping_greater = mapping_sequence<'>'>;
using mapping_plus = mapping_sequence<'+'>;
using mapping_minus = mapping_sequence<'-'>;
using mapping_tilde = mapping_sequence<'~'>;
using mapping_star = mapping_sequence<'*'>;
using mapping_slash = mapping_sequence<'/'>;
using mapping_percent = mapping_sequence<'%'>;
using mapping_equals = mapping_sequence<'='>;

struct mapping_close_expression_blocked_by_curly_depth {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return mapping_close_expression{}(ev, ctx) &&
           ev.request.cursor.curly_bracket_depth > 0u;
  }
};

struct mapping_close_expression_not_blocked {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return mapping_close_expression{}(ev, ctx) &&
           !mapping_close_expression_blocked_by_curly_depth{}(ev, ctx);
  }
};

struct starts_string {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    if (ev.ctx.pos >= ev.ctx.size) {
      return false;
    }
    const char c = ev.ctx.source[ev.ctx.pos];
    return c == '\'' || c == '"';
  }
};

struct starts_numeric {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size &&
           ::emel::text::jinja::lexer::detail::is_integer(ev.ctx.source[ev.ctx.pos]);
  }
};

struct starts_word {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size &&
           ::emel::text::jinja::lexer::detail::is_word(ev.ctx.source[ev.ctx.pos]);
  }
};

struct string_terminated {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.pos < ev.ctx.size;
  }
};

struct string_not_terminated {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.pos >= ev.ctx.size;
  }
};

} // namespace emel::text::jinja::parser::lexer::guard
