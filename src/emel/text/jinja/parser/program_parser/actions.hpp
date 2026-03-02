#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/context.hpp"
#include "emel/text/jinja/parser/program_parser/errors.hpp"

namespace emel::text::jinja::parser::program_parser::action {

inline const emel::text::jinja::token &
current_token(const event::parse_runtime &ev) noexcept {
  return ev.ctx.lex_result.tokens[ev.ctx.token_index];
}

template <class node_type, class... args>
inline emel::text::jinja::ast_ptr make_node(const size_t pos,
                                            args &&...args_in) {
  auto node = std::make_unique<node_type>(std::forward<args>(args_in)...);
  node->pos = pos;
  return node;
}

struct start_program_parse {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.phase = event::parse_phase::parsing;
    ev.ctx.statement = event::statement_kind::unknown;
    ev.ctx.expression = event::expression_kind::unknown;
    ev.ctx.token_index = 0;
    ev.ctx.statement_start = 0;
    ev.ctx.expression_start = 0;
    ev.ctx.expression_value_index = 0;
  }
};

struct consume_text {
  void operator()(const event::parse_runtime &ev, context &) const {
    const auto &tok = current_token(ev);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::string_literal>(tok.pos, tok.value));
    ++ev.ctx.token_index;
  }
};

struct consume_comment {
  void operator()(const event::parse_runtime &ev, context &) const {
    const auto &tok = current_token(ev);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::comment_statement>(tok.pos, tok.value));
    ++ev.ctx.token_index;
  }
};

struct finish_parsed {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_done(ev.request, ev.ctx);
  }
};

struct fail_current_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::parse_failed, current_token(ev).pos);
  }
};

struct on_unexpected {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::internal_error, ev.ctx.error_pos_out);
  }

  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr start_program_parse start_program_parse{};
inline constexpr consume_text consume_text{};
inline constexpr consume_comment consume_comment{};
inline constexpr finish_parsed finish_parsed{};
inline constexpr fail_current_token fail_current_token{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::jinja::parser::program_parser::action
