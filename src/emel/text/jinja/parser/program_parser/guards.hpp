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

inline error runtime_error(const event::parse_runtime &ev) noexcept {
  return ev.ctx.err;
}

inline bool error_is(const error runtime_err,
                     const error expected) noexcept {
  return runtime_err == expected;
}

inline bool error_is_unknown(const error runtime_err) noexcept {
  return !error_is(runtime_err, error::none) &&
         !error_is(runtime_err, error::invalid_request) &&
         !error_is(runtime_err, error::parse_failed) &&
         !error_is(runtime_err, error::internal_error) &&
         !error_is(runtime_err, error::untracked);
}

struct parse_error_none {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct parse_error_invalid_request {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct parse_error_parse_failed {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct parse_error_internal_error {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct parse_error_untracked {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct parse_error_unknown {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
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
    return has_token(ev.ctx) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::text) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::comment) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::open_expression) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::open_statement);
  }
};

} // namespace emel::text::jinja::parser::program_parser::guard
