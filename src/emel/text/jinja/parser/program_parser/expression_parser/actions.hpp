#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/context.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/errors.hpp"

namespace emel::text::jinja::parser::program_parser::expression_parser::action {

inline const emel::text::jinja::token &
current_token(const event::parse_runtime &ev) noexcept {
  return ev.ctx.lex_result.tokens[ev.ctx.token_index];
}

inline const emel::text::jinja::token &
token_at_index(const event::parse_runtime &ev, const size_t index) noexcept {
  return ev.ctx.lex_result.tokens[index];
}

template <class node_type, class... args>
inline emel::text::jinja::ast_ptr make_node(const size_t pos,
                                            args &&...args_in) {
  auto node = std::make_unique<node_type>(std::forward<args>(args_in)...);
  node->pos = pos;
  return node;
}

struct begin_expression_parse {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::expression;
    ev.ctx.expression = event::expression_kind::unknown;
    ev.ctx.expression_start = ev.ctx.token_index;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    ++ev.ctx.token_index;
  }
};

struct consume_expression_identifier {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::identifier;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    ++ev.ctx.token_index;
  }
};

struct consume_expression_identifier_and_close {
  void operator()(const event::parse_runtime &ev, context &) const {
    ev.ctx.expression = event::expression_kind::identifier;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    const auto &tok = current_token(ev);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::identifier>(tok.pos, tok.value));
    ev.ctx.token_index += 2u;
  }
};

struct consume_expression_literal {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::literal;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    ++ev.ctx.token_index;
  }
};

struct consume_expression_unary {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::unary;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    ++ev.ctx.token_index;
  }
};

struct consume_expression_compound {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::compound;
    ev.ctx.expression_value_index = ev.ctx.token_index;
    ++ev.ctx.token_index;
  }
};

struct consume_expression_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ++ev.ctx.token_index;
  }
};

struct emit_expression_identifier {
  void operator()(const event::parse_runtime &ev, context &) const {
    const auto &tok = token_at_index(ev, ev.ctx.expression_value_index);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::identifier>(tok.pos, tok.value));
  }
};

struct emit_expression_generic {
  void operator()(const event::parse_runtime &ev, context &) const {
    const auto &tok = token_at_index(ev, ev.ctx.expression_value_index);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::string_literal>(tok.pos, tok.value));
  }
};

struct consume_expression_close {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ++ev.ctx.token_index;
  }
};

struct fail_expression_close_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::parse_failed, current_token(ev).pos);
  }
};

struct fail_expression_start_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    const auto &tok = token_at_index(ev, ev.ctx.expression_start);
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::parse_failed, tok.pos);
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

inline constexpr begin_expression_parse begin_expression_parse{};
inline constexpr consume_expression_identifier consume_expression_identifier{};
inline constexpr consume_expression_identifier_and_close consume_expression_identifier_and_close{};
inline constexpr consume_expression_literal consume_expression_literal{};
inline constexpr consume_expression_unary consume_expression_unary{};
inline constexpr consume_expression_compound consume_expression_compound{};
inline constexpr consume_expression_token consume_expression_token{};
inline constexpr emit_expression_identifier emit_expression_identifier{};
inline constexpr emit_expression_generic emit_expression_generic{};
inline constexpr consume_expression_close consume_expression_close{};
inline constexpr fail_expression_close_token fail_expression_close_token{};
inline constexpr fail_expression_start_token fail_expression_start_token{};
inline constexpr on_unexpected on_unexpected{};

} // namespace
  // emel::text::jinja::parser::program_parser::expression_parser::action
