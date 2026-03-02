#pragma once

#include <cstddef>

#include "emel/text/jinja/parser/classifier_parser/context.hpp"
#include "emel/text/jinja/parser/classifier_parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::classifier_parser::guard {

inline bool has_token(const event::parse_ctx & ctx,
                      const size_t offset = 0) noexcept {
  return ctx.token_index + offset < ctx.lex_result.tokens.size();
}

inline bool token_is(const event::parse_ctx & ctx,
                     const emel::text::jinja::token_type type,
                     const size_t offset = 0) noexcept {
  return has_token(ctx, offset) &&
         ctx.lex_result.tokens[ctx.token_index + offset].type == type;
}

struct no_tokens {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return !has_token(ev.ctx);
  }
};

struct token_text {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::text);
  }
};

struct token_comment {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::comment);
  }
};

struct token_open_expression {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::open_expression);
  }
};

struct token_open_statement {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::open_statement);
  }
};

struct token_unknown {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return has_token(ev.ctx) &&
           !token_text{}(ev, action::context{}) &&
           !token_comment{}(ev, action::context{}) &&
           !token_open_expression{}(ev, action::context{}) &&
           !token_open_statement{}(ev, action::context{});
  }
};

struct statement_expression {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.statement == event::statement_kind::expression;
  }
};

struct statement_not_expression {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.statement != event::statement_kind::expression;
  }
};

struct expr_no_token {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return !has_token(ev.ctx, 1);
  }
};

struct expr_token_literal {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::numeric_literal, 1) ||
           token_is(ev.ctx, emel::text::jinja::token_type::string_literal, 1) ||
           token_is(ev.ctx, emel::text::jinja::token_type::open_square_bracket, 1) ||
           token_is(ev.ctx, emel::text::jinja::token_type::open_curly_bracket, 1);
  }
};

struct expr_token_identifier {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::identifier, 1);
  }
};

struct expr_token_unary {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::additive_binary_operator, 1) ||
           token_is(ev.ctx, emel::text::jinja::token_type::unary_operator, 1);
  }
};

struct expr_token_compound {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::open_paren, 1);
  }
};

struct expr_token_unknown {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return has_token(ev.ctx, 1) &&
           !expr_token_literal{}(ev, action::context{}) &&
           !expr_token_identifier{}(ev, action::context{}) &&
           !expr_token_unary{}(ev, action::context{}) &&
           !expr_token_compound{}(ev, action::context{});
  }
};

struct phase_ok {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == error::none;
  }
};

struct phase_failed {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err != error::none;
  }
};

}  // namespace emel::text::jinja::parser::classifier_parser::guard
