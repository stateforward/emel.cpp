#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/context.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/errors.hpp"

namespace emel::text::jinja::parser::program_parser::statement_parser::action {

inline const emel::text::jinja::token &
current_token(const event::parse_runtime &ev) noexcept {
  return ev.ctx.lex_result.tokens[ev.ctx.token_index];
}

inline const emel::text::jinja::token &token_at(const event::parse_runtime &ev,
                                                const size_t offset) noexcept {
  return ev.ctx.lex_result.tokens[ev.ctx.token_index + offset];
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

struct begin_statement_scan {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::statement;
    ev.ctx.statement_start = ev.ctx.token_index;
    ev.ctx.token_index += 2;
  }
};

struct consume_statement_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ++ev.ctx.token_index;
  }
};

struct consume_statement_close_and_emit {
  void operator()(const event::parse_runtime &ev, context &) const {
    const auto &tok = token_at_index(ev, ev.ctx.statement_start);
    ev.request.program.body.push_back(
        make_node<emel::text::jinja::noop_statement>(tok.pos));
    ++ev.ctx.token_index;
  }
};

struct fail_statement_name_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::parse_failed, token_at(ev, 1).pos);
  }
};

struct fail_statement_open_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::parse_failed, current_token(ev).pos);
  }
};

struct fail_statement_start_token {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    const auto &tok = token_at_index(ev, ev.ctx.statement_start);
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

inline constexpr begin_statement_scan begin_statement_scan{};
inline constexpr consume_statement_token consume_statement_token{};
inline constexpr consume_statement_close_and_emit
    consume_statement_close_and_emit{};
inline constexpr fail_statement_name_token fail_statement_name_token{};
inline constexpr fail_statement_open_token fail_statement_open_token{};
inline constexpr fail_statement_start_token fail_statement_start_token{};
inline constexpr on_unexpected on_unexpected{};

} // namespace
  // emel::text::jinja::parser::program_parser::statement_parser::action
