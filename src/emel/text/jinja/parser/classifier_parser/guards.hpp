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

inline bool parse_error_is(const event::parse_runtime & ev, const error code_value) noexcept {
  return ev.ctx.err == code_value;
}

struct parse_error_none {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return parse_error_is(ev, error::none);
  }
};

struct parse_error_invalid_request {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return parse_error_is(ev, error::invalid_request);
  }
};

struct parse_error_parse_failed {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return parse_error_is(ev, error::parse_failed);
  }
};

struct parse_error_internal_error {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return parse_error_is(ev, error::internal_error);
  }
};

struct parse_error_untracked {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return parse_error_is(ev, error::untracked);
  }
};

}  // namespace emel::text::jinja::parser::classifier_parser::guard
