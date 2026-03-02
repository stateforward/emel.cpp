#pragma once

#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/context.hpp"
#include "emel/text/jinja/parser/program_parser/errors.hpp"

namespace emel::text::jinja::parser::program_parser::guard {

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

struct phase_ok {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == error::none;
  }
};

struct phase_failed {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != error::none;
  }
};

struct at_eof {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return !has_token(ev.ctx);
  }
};

struct token_text {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::text);
  }
};

struct token_comment {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::comment);
  }
};

struct token_open_expression {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::open_expression);
  }
};

struct token_open_statement {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::open_statement);
  }
};

struct token_unexpected {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return has_token(ev.ctx) && !token_text{}(ev, action::context{}) &&
           !token_comment{}(ev, action::context{}) &&
           !token_open_expression{}(ev, action::context{}) &&
           !token_open_statement{}(ev, action::context{});
  }
};

} // namespace emel::text::jinja::parser::program_parser::guard
