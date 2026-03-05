#pragma once

#include <string_view>
#include <utility>

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
                  const action::context &ctx) const noexcept {
    (void)ctx;
    return ev.request.cursor.source.data() != nullptr &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error) &&
           ev.request.cursor.offset > ev.request.cursor.source.size();
  }
};

inline bool scan_error_is(const event::next_runtime &ev,
                          const error expected) noexcept {
  return ev.ctx.handled && ev.ctx.scan.err == detail::error_code(expected);
}

inline bool scan_error_is_unknown(const event::next_runtime &ev) noexcept {
  return ev.ctx.handled && ev.ctx.scan.err != detail::error_code(error::none) &&
         ev.ctx.scan.err != detail::error_code(error::invalid_request) &&
         ev.ctx.scan.err != detail::error_code(error::parse_failed) &&
         ev.ctx.scan.err != detail::error_code(error::internal_error) &&
         ev.ctx.scan.err != detail::error_code(error::untracked);
}

struct parse_error_none {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::none);
  }
};

struct parse_error_invalid_request {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::invalid_request);
  }
};

struct parse_error_parse_failed {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::parse_failed);
  }
};

struct parse_error_internal_error {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::internal_error);
  }
};

struct parse_error_untracked {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::untracked);
  }
};

struct parse_error_unknown {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is_unknown(ev);
  }
};

struct scan_token_available {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::none) && ev.ctx.scan.has_token;
  }
};

struct scan_no_token_eof {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return scan_error_is(ev, error::none) && !ev.ctx.scan.has_token;
  }
};

struct scan_unhandled {
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

struct text_token_empty_at_end {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return text_token_empty{}(ev, ctx) && ev.ctx.pos >= ev.ctx.size;
  }
};

struct text_boundary_empty_at_end {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.text_start == ev.ctx.text_end &&
           ev.ctx.pos >= ev.ctx.size;
  }
};

struct text_plain_boundary_ready {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    const bool has_text = ev.ctx.text_end > ev.ctx.text_start;
    const bool can_trim_leading_newline =
        has_text && ev.request.cursor.last_block_can_trim_newline &&
        ev.ctx.source[ev.ctx.text_start] == '\n';
    const bool next_block_lstrip_marker_present =
        ev.ctx.pos < ev.ctx.size &&
        ev.ctx.source[ev.ctx.pos] == '{' &&
        ::emel::text::jinja::lexer::detail::next_pos_is<'{', '%', '#'>(
            ev.ctx.source, ev.ctx.pos) &&
        ::emel::text::jinja::lexer::detail::next_pos_is<'-'>(
            ev.ctx.source, ev.ctx.pos, 2u);
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           has_text &&
           !can_trim_leading_newline &&
           !ev.request.cursor.last_block_rstrip &&
           !next_block_lstrip_marker_present;
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
           ::emel::text::jinja::lexer::detail::next_pos_is<'%', '#', '-'>(
               ev.ctx.source, ev.ctx.pos);
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

struct text_opening_trim_stopped_on_newline {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.text_trim_probe > ev.ctx.text_start &&
           ev.ctx.source[ev.ctx.text_trim_probe - 1u] == '\n';
  }
};

struct text_opening_trim_to_zero {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           ev.ctx.text_start == 0u &&
           ev.ctx.text_trim_probe == 0u;
  }
};

struct text_opening_trim_keep_original {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !text_opening_trim_stopped_on_newline{}(ev, ctx) &&
           !text_opening_trim_to_zero{}(ev, ctx) &&
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
           ::emel::text::jinja::lexer::detail::next_pos_is<'{', '%', '#'>(
               ev.ctx.source, ev.ctx.pos) &&
           ::emel::text::jinja::lexer::detail::next_pos_is<'-'>(
               ev.ctx.source, ev.ctx.pos, 2u);
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
           ::emel::text::jinja::lexer::detail::next_pos_is<'#'>(ev.ctx.source, ev.ctx.pos);
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
           ::emel::text::jinja::lexer::detail::next_pos_is<'}'>(ev.ctx.source, ev.ctx.pos);
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
    if (ev.ctx.pos >= ev.ctx.size) {
      return false;
    }

    const size_t pos = ev.ctx.pos;
    const char ch = ev.ctx.source[pos];
    if (ch == '+') {
      return true;
    }
    if (ch != '-') {
      return false;
    }

    return pos + 1u >= ev.ctx.size ||
           (ev.ctx.source[pos + 1u] != '%' && ev.ctx.source[pos + 1u] != '}');
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
  template <size_t... indices>
  static bool matches(const char *cursor, std::index_sequence<indices...>) noexcept {
    constexpr char seq[] = {seq_chars...};
    return ((cursor[indices] == seq[indices]) && ...);
  }

  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    constexpr size_t seq_size = sizeof...(seq_chars);
    if (ev.ctx.pos + seq_size > ev.ctx.size) {
      return false;
    }
    return matches(ev.ctx.source.data() + ev.ctx.pos,
                   std::make_index_sequence<seq_size>{});
  }
};

template <char ch>
struct mapping_current_char {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.pos < ev.ctx.size && ev.ctx.source[ev.ctx.pos] == ch;
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
           ev.request.cursor.curly_bracket_depth == 0u;
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

struct string_scan_immediate_termination_or_eof {
  bool operator()(const event::next_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none) &&
           (ev.ctx.pos >= ev.ctx.size || ev.ctx.source[ev.ctx.pos] == ev.ctx.string_terminal);
  }
};

struct string_scan_requires_content {
  bool operator()(const event::next_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !string_scan_immediate_termination_or_eof{}(ev, ctx) &&
           ev.ctx.handled &&
           ev.ctx.scan.err == detail::error_code(error::none);
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
