#pragma once

#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/classifier_parser/context.hpp"
#include "emel/text/jinja/parser/classifier_parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::classifier_parser::action {

struct begin_classification {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.phase = event::parse_phase::statement_classification;
    ev.ctx.statement = event::statement_kind::unknown;
    ev.ctx.expression = event::expression_kind::unknown;
    ev.ctx.token_index = 0;
  }
};

struct set_statement_text {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::text;
    ev.ctx.expression = event::expression_kind::unknown;
  }
};

struct set_statement_comment {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::comment;
    ev.ctx.expression = event::expression_kind::unknown;
  }
};

struct set_statement_expression {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::expression;
  }
};

struct set_statement_statement {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::statement;
    ev.ctx.expression = event::expression_kind::unknown;
  }
};

struct set_statement_unknown {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.statement = event::statement_kind::unknown;
    ev.ctx.expression = event::expression_kind::unknown;
  }
};

struct set_expression_literal {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::literal;
  }
};

struct set_expression_identifier {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::identifier;
  }
};

struct set_expression_unary {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::unary;
  }
};

struct set_expression_compound {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::compound;
  }
};

struct set_expression_unknown {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.expression = event::expression_kind::unknown;
  }
};

struct on_unexpected {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ::emel::text::jinja::parser::action::helper::mark_error(
        ev.request, ev.ctx, error::internal_error, ev.ctx.error_pos_out);
  }

  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr begin_classification begin_classification{};
inline constexpr set_statement_text set_statement_text{};
inline constexpr set_statement_comment set_statement_comment{};
inline constexpr set_statement_expression set_statement_expression{};
inline constexpr set_statement_statement set_statement_statement{};
inline constexpr set_statement_unknown set_statement_unknown{};
inline constexpr set_expression_literal set_expression_literal{};
inline constexpr set_expression_identifier set_expression_identifier{};
inline constexpr set_expression_unary set_expression_unary{};
inline constexpr set_expression_compound set_expression_compound{};
inline constexpr set_expression_unknown set_expression_unknown{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::jinja::parser::classifier_parser::action
