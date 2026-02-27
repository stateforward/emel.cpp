#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/expression_parser/events.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/term_parser/events.hpp"

namespace emel::gbnf::rule_parser::events {

struct parsing_done;
struct parsing_error;

}  // namespace emel::gbnf::rule_parser::events

namespace emel::gbnf::rule_parser::event {

struct parse {
  std::string_view grammar_text = {};
  emel::gbnf::grammar * grammar_out = nullptr;
  ::emel::callback<bool(const ::emel::gbnf::rule_parser::events::parsing_done &)> dispatch_done = {};
  ::emel::callback<bool(const ::emel::gbnf::rule_parser::events::parsing_error &)> dispatch_error = {};
};

// Internal context object carried via completion<parse_rules>.
struct parse_rules_ctx {
  enum class term_origin : uint8_t {
    none = 0,
    need_term = 1,
    after_term = 2,
  };

  emel::gbnf::rule_parser::lexer::cursor cursor = {};
  emel::gbnf::rule_parser::lexer::event::token token = {};
  emel::gbnf::rule_parser::nonterm_parser::events::parse_mode nonterm_mode =
      emel::gbnf::rule_parser::nonterm_parser::events::parse_mode::none;
  emel::gbnf::rule_parser::expression_parser::events::parse_kind expression_kind =
      emel::gbnf::rule_parser::expression_parser::events::parse_kind::unknown;
  emel::gbnf::rule_parser::term_parser::events::term_kind term_kind =
      emel::gbnf::rule_parser::term_parser::events::term_kind::unknown;
  term_origin current_term_origin = term_origin::none;
  uint32_t nonterm_rule_id = 0;
  bool has_token = false;
  emel::error::type err = emel::error::cast(error::none);
};

// Internal event used by rule_parser::sm wrapper; not part of public API.
struct parse_rules {
  const parse & request;
  parse_rules_ctx & ctx;
};

}  // namespace emel::gbnf::rule_parser::event

namespace emel::gbnf::rule_parser::events {

struct parsing_done {
  emel::gbnf::grammar & grammar;
};

struct parsing_error {
  emel::gbnf::grammar & grammar;
  int32_t err = 0;
};

}  // namespace emel::gbnf::rule_parser::events
