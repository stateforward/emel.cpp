#pragma once

#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/context.hpp"

namespace emel::text::jinja::parser::program_parser::expression_parser::guard {

inline bool has_token(const event::parse_ctx &ctx,
                      const size_t offset = 0) noexcept {
  return ctx.token_index + offset < ctx.lex_result.tokens.size();
}

inline const emel::text::jinja::token &
token_at(const event::parse_ctx &ctx, const size_t offset = 0) noexcept {
  return ctx.lex_result.tokens[ctx.token_index + offset];
}

inline bool token_is(const event::parse_ctx &ctx,
                     const emel::text::jinja::token_type type,
                     const size_t offset = 0) noexcept {
  return has_token(ctx, offset) && token_at(ctx, offset).type == type;
}

struct expression_identifier {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.expression == event::expression_kind::identifier;
  }
};

struct expression_non_identifier {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.expression != event::expression_kind::identifier;
  }
};

struct expr_first_is_close {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::close_expression);
  }
};

struct expr_first_is_identifier {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::identifier);
  }
};

struct expr_first_is_literal {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::numeric_literal) ||
           token_is(ev.ctx, emel::text::jinja::token_type::string_literal) ||
           token_is(ev.ctx,
                    emel::text::jinja::token_type::open_square_bracket) ||
           token_is(ev.ctx, emel::text::jinja::token_type::open_curly_bracket);
  }
};

struct expr_first_is_unary {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx,
                    emel::text::jinja::token_type::additive_binary_operator) ||
           token_is(ev.ctx, emel::text::jinja::token_type::unary_operator);
  }
};

struct expr_first_is_other_content {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return has_token(ev.ctx) && !expr_first_is_close{}(ev, action::context{}) &&
           !expr_first_is_identifier{}(ev, action::context{}) &&
           !expr_first_is_literal{}(ev, action::context{}) &&
           !expr_first_is_unary{}(ev, action::context{});
  }
};

struct expr_scan_at_close {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::close_expression);
  }
};

struct expr_scan_continue {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return has_token(ev.ctx) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::close_expression);
  }
};

struct expr_scan_eof {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return !has_token(ev.ctx);
  }
};

} // namespace
  // emel::text::jinja::parser::program_parser::expression_parser::guard
