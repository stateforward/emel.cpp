#pragma once

#include <string_view>

#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/context.hpp"

namespace emel::text::jinja::parser::program_parser::statement_parser::guard {

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

inline bool statement_name_is(const event::parse_ctx &ctx,
                              const std::string_view expected) noexcept {
  return token_is(ctx, emel::text::jinja::token_type::identifier, 1) &&
         token_at(ctx, 1).value == expected;
}

struct statement_identifier_missing {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return !token_is(ev.ctx, emel::text::jinja::token_type::identifier, 1);
  }
};

struct statement_name_set {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "set");
  }
};

struct statement_name_if {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "if");
  }
};

struct statement_name_elif {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "elif");
  }
};

struct statement_name_else {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "else");
  }
};

struct statement_name_endif {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endif");
  }
};

struct statement_name_for {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "for");
  }
};

struct statement_name_endfor {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endfor");
  }
};

struct statement_name_macro {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "macro");
  }
};

struct statement_name_endmacro {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endmacro");
  }
};

struct statement_name_call {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "call");
  }
};

struct statement_name_endcall {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endcall");
  }
};

struct statement_name_filter {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "filter");
  }
};

struct statement_name_endfilter {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endfilter");
  }
};

struct statement_name_break {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "break");
  }
};

struct statement_name_continue {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "continue");
  }
};

struct statement_name_generation {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "generation");
  }
};

struct statement_name_endgeneration {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endgeneration");
  }
};

struct statement_name_endset {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return statement_name_is(ev.ctx, "endset");
  }
};

struct statement_name_unknown {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::identifier, 1) &&
           !statement_name_set{}(ev, action::context{}) &&
           !statement_name_if{}(ev, action::context{}) &&
           !statement_name_elif{}(ev, action::context{}) &&
           !statement_name_else{}(ev, action::context{}) &&
           !statement_name_endif{}(ev, action::context{}) &&
           !statement_name_for{}(ev, action::context{}) &&
           !statement_name_endfor{}(ev, action::context{}) &&
           !statement_name_macro{}(ev, action::context{}) &&
           !statement_name_endmacro{}(ev, action::context{}) &&
           !statement_name_call{}(ev, action::context{}) &&
           !statement_name_endcall{}(ev, action::context{}) &&
           !statement_name_filter{}(ev, action::context{}) &&
           !statement_name_endfilter{}(ev, action::context{}) &&
           !statement_name_break{}(ev, action::context{}) &&
           !statement_name_continue{}(ev, action::context{}) &&
           !statement_name_generation{}(ev, action::context{}) &&
           !statement_name_endgeneration{}(ev, action::context{}) &&
           !statement_name_endset{}(ev, action::context{});
  }
};

struct statement_scan_at_close {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return token_is(ev.ctx, emel::text::jinja::token_type::close_statement);
  }
};

struct statement_scan_continue {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return has_token(ev.ctx) &&
           !token_is(ev.ctx, emel::text::jinja::token_type::close_statement);
  }
};

struct statement_scan_eof {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return !has_token(ev.ctx);
  }
};

} // namespace
  // emel::text::jinja::parser::program_parser::statement_parser::guard
