#pragma once

#include <string_view>

#include "emel/gbnf/rule_parser/expression_parser/events.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/term_parser/events.hpp"

namespace emel::gbnf::rule_parser::guard {

template <lexer::event::token_kind kind>
struct lexer_token_is {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.has_token &&
           ev.ctx.token.kind == kind;
  }
};

inline bool current_rule_has_space(const action::context & ctx, const uint32_t count) noexcept {
  return ctx.current_rule.size + count <= emel::gbnf::k_max_gbnf_rule_elements;
}

inline bool can_finalize_active_rule(const event::parse_rules & ev,
                                     const action::context & ctx) noexcept {
  if (ctx.group_depth != 0u || ctx.current_rule.size == 0u) {
    return false;
  }
  if (!current_rule_has_space(ctx, 1u)) {
    return false;
  }
  const uint32_t rule_id = ctx.current_rule_id;
  const emel::gbnf::grammar & grammar = *ev.request.grammar_out;
  if (rule_id >= emel::gbnf::k_max_gbnf_rules || grammar.rule_lengths[rule_id] != 0u) {
    return false;
  }
  return grammar.element_count + ctx.current_rule.size + 1u <= emel::gbnf::k_max_gbnf_elements;
}

inline bool can_finalize_symbols(const event::parse_rules & ev,
                                 const action::context & ctx) noexcept {
  const emel::gbnf::grammar & grammar = *ev.request.grammar_out;
  if (grammar.rule_count == 0u) {
    return false;
  }
  for (uint32_t id = 0; id < ctx.next_symbol_id; ++id) {
    if (!ctx.rule_defined[id]) {
      return false;
    }
  }
  return true;
}

inline bool literal_element_count(const std::string_view text, uint32_t & count) noexcept {
  if (text.size() < 2u || text.front() != '"' || text.back() != '"') {
    return false;
  }
  count = 0;
  const char * pos = text.data() + 1u;
  const char * end = text.data() + text.size() - 1u;
  while (pos < end) {
    const auto parsed = emel::gbnf::rule_parser::detail::parse_char(pos, end);
    if (parsed.second == nullptr) {
      return false;
    }
    ++count;
    pos = parsed.second;
  }
  return true;
}

inline bool character_class_element_count(const std::string_view text, uint32_t & count) noexcept {
  if (text.size() < 2u || text.front() != '[' || text.back() != ']') {
    return false;
  }
  count = 0;
  const char * pos = text.data() + 1u;
  const char * end = text.data() + text.size() - 1u;
  if (pos < end && *pos == '^') {
    ++pos;
  }

  bool first = true;
  while (pos < end) {
    const auto first_char = emel::gbnf::rule_parser::detail::parse_char(pos, end);
    if (first_char.second == nullptr) {
      return false;
    }
    ++count;
    first = false;
    pos = first_char.second;

    if (pos + 1u < end && pos[0] == '-' && pos[1] != ']') {
      ++pos;
      const auto range_char = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      if (range_char.second == nullptr) {
        return false;
      }
      ++count;
      pos = range_char.second;
    }
  }
  return !first;
}

struct valid_parse {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.request.grammar_text.data() != nullptr &&
           !ev.request.grammar_text.empty() &&
           ev.request.grammar_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_parse {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return !valid_parse{}(ev, ctx);
  }
};

struct invalid_parse_with_dispatchable_grammar {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return invalid_parse{}(ev, ctx) &&
           ev.request.grammar_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_parse_with_grammar_only {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return invalid_parse{}(ev, ctx) &&
           ev.request.grammar_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_parse_without_grammar {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return invalid_parse{}(ev, ctx) && ev.request.grammar_out == nullptr;
  }
};

struct parse_error_none {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct parse_error_invalid_request {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::invalid_request);
  }
};

struct parse_error_parse_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::parse_failed);
  }
};

struct parse_error_internal_error {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::internal_error);
  }
};

struct parse_error_untracked {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::untracked);
  }
};

struct parse_error_unknown {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) &&
           ev.ctx.err != emel::error::cast(error::invalid_request) &&
           ev.ctx.err != emel::error::cast(error::parse_failed) &&
           ev.ctx.err != emel::error::cast(error::internal_error) &&
           ev.ctx.err != emel::error::cast(error::untracked);
  }
};

struct lexer_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct lexer_at_eof {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) && !ev.ctx.has_token;
  }
};

struct lexer_has_token {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) && ev.ctx.has_token;
  }
};

struct definition_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct definition_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct expression_done_identifier {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.expression_kind == expression_parser::events::parse_kind::identifier;
  }
};

struct expression_done_non_identifier {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.expression_kind == expression_parser::events::parse_kind::non_identifier;
  }
};

struct expression_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct nonterm_definition_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.nonterm_mode == nonterm_parser::events::parse_mode::definition;
  }
};

struct nonterm_reference_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.nonterm_mode == nonterm_parser::events::parse_mode::reference;
  }
};

struct nonterm_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct term_from_need_term {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.current_term_origin == event::parse_rules_ctx::term_origin::need_term;
  }
};

struct term_from_after_term {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.current_term_origin == event::parse_rules_ctx::term_origin::after_term;
  }
};

struct term_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

template <term_parser::events::term_kind kind>
struct term_kind_is {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) && ev.ctx.term_kind == kind;
  }
};

struct token_identifier {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return lexer_token_is<lexer::event::token_kind::identifier>{}(ev, ctx);
  }
};

struct token_newline {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return lexer_token_is<lexer::event::token_kind::newline>{}(ev, ctx) ||
           term_kind_is<term_parser::events::term_kind::newline>{}(ev, ctx);
  }
};

struct token_alternation_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::alternation>{}(ev, ctx) &&
           current_rule_has_space(ctx, 1u);
  }
};

struct token_literal_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    if (!term_kind_is<term_parser::events::term_kind::string_literal>{}(ev, ctx)) {
      return false;
    }
    uint32_t count = 0;
    if (!literal_element_count(ev.ctx.token.text, count)) {
      return false;
    }
    return current_rule_has_space(ctx, count);
  }
};

struct token_character_class_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    if (!term_kind_is<term_parser::events::term_kind::character_class>{}(ev, ctx)) {
      return false;
    }
    uint32_t count = 0;
    if (!character_class_element_count(ev.ctx.token.text, count)) {
      return false;
    }
    return current_rule_has_space(ctx, count);
  }
};

struct token_rule_reference_candidate {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::rule_reference>{}(ev, ctx) &&
           current_rule_has_space(ctx, 1u);
  }
};

struct rule_reference_token_negated_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return token_rule_reference_candidate{}(ev, ctx) &&
           ev.ctx.token.text.size() >= 1u &&
           ev.ctx.token.text.front() == '!';
  }
};

struct rule_reference_token_plain_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return token_rule_reference_candidate{}(ev, ctx) &&
           !rule_reference_token_negated_shape{}(ev, ctx);
  }
};

struct rule_reference_plain_envelope_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return rule_reference_token_plain_shape{}(ev, ctx) &&
           ev.ctx.token.text.size() >= 4u &&
           ev.ctx.token.text[0] == '<' &&
           ev.ctx.token.text[1] == '[' &&
           ev.ctx.token.text[ev.ctx.token.text.size() - 2u] == ']' &&
           ev.ctx.token.text[ev.ctx.token.text.size() - 1u] == '>';
  }
};

struct rule_reference_plain_envelope_invalid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return rule_reference_token_plain_shape{}(ev, ctx) &&
           !rule_reference_plain_envelope_valid{}(ev, ctx);
  }
};

struct rule_reference_negated_envelope_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return rule_reference_token_negated_shape{}(ev, ctx) &&
           ev.ctx.token.text.size() >= 5u &&
           ev.ctx.token.text[0] == '!' &&
           ev.ctx.token.text[1] == '<' &&
           ev.ctx.token.text[2] == '[' &&
           ev.ctx.token.text[ev.ctx.token.text.size() - 2u] == ']' &&
           ev.ctx.token.text[ev.ctx.token.text.size() - 1u] == '>';
  }
};

struct rule_reference_negated_envelope_invalid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return rule_reference_token_negated_shape{}(ev, ctx) &&
           !rule_reference_negated_envelope_valid{}(ev, ctx);
  }
};

struct token_newline_with_group_depth_zero_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return token_newline{}(ev, ctx) &&
           ctx.group_depth == 0u &&
           can_finalize_active_rule(ev, ctx);
  }
};

struct token_newline_with_group_depth_nonzero {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return token_newline{}(ev, ctx) && ctx.group_depth != 0u;
  }
};

struct token_dot_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::dot>{}(ev, ctx) &&
           current_rule_has_space(ctx, 1u);
  }
};

struct token_open_group_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::open_group>{}(ev, ctx) &&
           ctx.group_depth < ctx.group_stack.size() &&
           ctx.next_symbol_id < emel::gbnf::k_max_gbnf_rules;
  }
};

struct token_close_group_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    if (!term_kind_is<term_parser::events::term_kind::close_group>{}(ev, ctx) ||
        ctx.group_depth == 0u) {
      return false;
    }

    const auto frame = ctx.group_stack[ctx.group_depth - 1u];
    if (frame.sequence_start > ctx.current_rule.size) {
      return false;
    }
    const uint32_t group_len = ctx.current_rule.size - frame.sequence_start;
    const uint32_t group_rule_size = group_len + 1u;
    const auto & grammar = *ev.request.grammar_out;
    if (group_rule_size > emel::gbnf::k_max_gbnf_rule_elements) {
      return false;
    }
    if (frame.generated_rule_id >= emel::gbnf::k_max_gbnf_rules ||
        grammar.rule_lengths[frame.generated_rule_id] != 0u) {
      return false;
    }
    if (grammar.element_count + group_rule_size > emel::gbnf::k_max_gbnf_elements) {
      return false;
    }
    return frame.sequence_start + 1u <= emel::gbnf::k_max_gbnf_rule_elements;
  }
};

struct quantifier_candidate {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::quantifier>{}(ev, ctx);
  }
};

struct quantifier_token_star {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_candidate{}(ev, ctx) && ev.ctx.token.text == "*";
  }
};

struct quantifier_token_plus {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_candidate{}(ev, ctx) && ev.ctx.token.text == "+";
  }
};

struct quantifier_token_question {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_candidate{}(ev, ctx) && ev.ctx.token.text == "?";
  }
};

struct quantifier_token_braced {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_candidate{}(ev, ctx) &&
           ev.ctx.token.text.size() >= 3u &&
           ev.ctx.token.text.front() == '{' &&
           ev.ctx.token.text.back() == '}';
  }
};

struct quantifier_braced_exact_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    const bool braced = quantifier_token_braced{}(ev, ctx);
    const std::string_view text = ev.ctx.token.text;
    const std::string_view core = text.substr(1u, text.size() - 2u);
    const size_t comma_pos = core.find(',');
    return braced && comma_pos == std::string_view::npos;
  }
};

struct quantifier_braced_open_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    const bool braced = quantifier_token_braced{}(ev, ctx);
    const std::string_view text = ev.ctx.token.text;
    const std::string_view core = text.substr(1u, text.size() - 2u);
    const size_t comma_pos = core.find(',');
    const bool has_comma = comma_pos != std::string_view::npos;
    const size_t suffix_offset = comma_pos + static_cast<size_t>(has_comma);
    return braced && has_comma && suffix_offset == core.size();
  }
};

struct quantifier_braced_range_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    const bool braced = quantifier_token_braced{}(ev, ctx);
    const std::string_view text = ev.ctx.token.text;
    const std::string_view core = text.substr(1u, text.size() - 2u);
    const size_t comma_pos = core.find(',');
    const bool has_comma = comma_pos != std::string_view::npos;
    const size_t suffix_offset = comma_pos + static_cast<size_t>(has_comma);
    return braced && has_comma && suffix_offset < core.size();
  }
};

struct quantifier_braced_invalid_shape {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_token_braced{}(ev, ctx) &&
           !quantifier_braced_exact_shape{}(ev, ctx) &&
           !quantifier_braced_open_shape{}(ev, ctx) &&
           !quantifier_braced_range_shape{}(ev, ctx);
  }
};

struct quantifier_token_unknown {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return quantifier_candidate{}(ev, ctx) &&
           !quantifier_token_star{}(ev, ctx) &&
           !quantifier_token_plus{}(ev, ctx) &&
           !quantifier_token_question{}(ev, ctx) &&
           !quantifier_token_braced{}(ev, ctx);
  }
};

struct term_need_literal_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) && token_literal_valid{}(ev, ctx);
  }
};

struct term_need_character_class_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) && token_character_class_valid{}(ev, ctx);
  }
};

struct term_need_rule_reference_candidate {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) &&
           term_kind_is<term_parser::events::term_kind::rule_reference>{}(ev, ctx);
  }
};

struct term_need_dot_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) && token_dot_valid{}(ev, ctx);
  }
};

struct term_need_open_group_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) && token_open_group_valid{}(ev, ctx);
  }
};

struct term_need_newline_with_group_depth_nonzero {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) &&
           token_newline_with_group_depth_nonzero{}(ev, ctx);
  }
};

struct term_after_literal_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_literal_valid{}(ev, ctx);
  }
};

struct term_after_character_class_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_character_class_valid{}(ev, ctx);
  }
};

struct term_after_rule_reference_candidate {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) &&
           term_kind_is<term_parser::events::term_kind::rule_reference>{}(ev, ctx);
  }
};

struct term_after_dot_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_dot_valid{}(ev, ctx);
  }
};

struct term_after_open_group_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_open_group_valid{}(ev, ctx);
  }
};

struct term_after_alternation_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_alternation_valid{}(ev, ctx);
  }
};

struct term_after_newline_with_group_depth_nonzero {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) &&
           token_newline_with_group_depth_nonzero{}(ev, ctx);
  }
};

struct term_after_newline_with_group_depth_zero_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) &&
           token_newline_with_group_depth_zero_valid{}(ev, ctx);
  }
};

struct term_after_close_group_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_close_group_valid{}(ev, ctx);
  }
};

struct term_after_quantifier_candidate {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && quantifier_candidate{}(ev, ctx);
  }
};

struct eof_can_finalize_active_rule {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return lexer_at_eof{}(ev, ctx) && can_finalize_active_rule(ev, ctx);
  }
};

struct eof_cannot_finalize_active_rule {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return lexer_at_eof{}(ev, ctx) && !can_finalize_active_rule(ev, ctx);
  }
};

struct eof_can_finalize_symbols {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return parse_error_none{}(ev, ctx) && can_finalize_symbols(ev, ctx);
  }
};

struct eof_cannot_finalize_symbols {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return parse_error_none{}(ev, ctx) && !can_finalize_symbols(ev, ctx);
  }
};

}  // namespace emel::gbnf::rule_parser::guard
