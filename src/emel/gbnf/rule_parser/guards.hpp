#pragma once

#include <limits>
#include <string_view>

#include "emel/gbnf/expression_parser/events.hpp"
#include "emel/gbnf/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/term_parser/events.hpp"

namespace emel::gbnf::rule_parser::guard {

template <lexer::event::token_kind kind>
struct lexer_token_is {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.has_token &&
           ev.flow.token.kind == kind;
  }
};

inline bool is_quantifier_text(const std::string_view text) noexcept {
  return text == "+" || text == "*" || text == "?" ||
         (text.size() >= 2u && text.front() == '{' && text.back() == '}');
}

inline bool parse_rule_reference_text(const std::string_view text) noexcept {
  std::size_t pos = 0;
  if (text.size() >= 1u && text[0] == '!') {
    pos = 1u;
  }
  if (text.size() < pos + 4u || text[pos] != '<' || text[pos + 1u] != '[') {
    return false;
  }
  pos += 2u;

  uint64_t value = 0;
  const char * cursor = text.data() + pos;
  const char * end = text.data() + text.size();
  const char * next = nullptr;
  if (!emel::gbnf::rule_parser::detail::parse_uint64(cursor, end, value, &next)) {
    return false;
  }
  pos = static_cast<std::size_t>(next - text.data());
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  return text.size() == pos + 2u && text[pos] == ']' && text[pos + 1u] == '>';
}

inline bool parse_quantifier_bounds(const std::string_view text,
                                    uint64_t & min_times,
                                    uint64_t & max_times) noexcept {
  constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
  if (text == "*") {
    min_times = 0;
    max_times = k_no_max;
    return true;
  }
  if (text == "+") {
    min_times = 1;
    max_times = k_no_max;
    return true;
  }
  if (text == "?") {
    min_times = 0;
    max_times = 1;
    return true;
  }
  if (text.size() < 3u || text.front() != '{' || text.back() != '}') {
    return false;
  }

  const char * cursor = text.data() + 1u;
  const char * end = text.data() + text.size() - 1u;
  const char * next = nullptr;
  if (!emel::gbnf::rule_parser::detail::parse_uint64(cursor, end, min_times, &next)) {
    return false;
  }
  if (next == end) {
    max_times = min_times;
    return true;
  }
  if (*next != ',') {
    return false;
  }
  ++next;
  if (next == end) {
    max_times = k_no_max;
    return true;
  }
  if (!emel::gbnf::rule_parser::detail::parse_uint64(next, end, max_times, &next)) {
    return false;
  }
  return next == end;
}

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

inline bool can_apply_quantifier(const event::parse_rules & ev,
                                 const action::context & ctx) noexcept {
  constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t k_max_repetition_threshold = 2000;
  if (ctx.last_sym_start == ctx.current_rule.size) {
    return false;
  }

  uint64_t min_times = 0;
  uint64_t max_times = 0;
  if (!parse_quantifier_bounds(ev.flow.token.text, min_times, max_times)) {
    return false;
  }
  if (min_times > k_max_repetition_threshold) {
    return false;
  }
  if (max_times != k_no_max && max_times > k_max_repetition_threshold) {
    return false;
  }
  if (max_times != k_no_max && max_times < min_times) {
    return false;
  }

  const uint64_t prev_len = static_cast<uint64_t>(ctx.current_rule.size - ctx.last_sym_start);
  const uint64_t repeated_len =
      min_times == 0 ? static_cast<uint64_t>(ctx.last_sym_start)
                     : static_cast<uint64_t>(ctx.last_sym_start) + prev_len * min_times;

  if (repeated_len > emel::gbnf::k_max_gbnf_rule_elements) {
    return false;
  }

  const bool no_max = max_times == k_no_max;
  const uint64_t n_opt = no_max ? 1 : (max_times - min_times);
  if (ctx.next_symbol_id + n_opt > emel::gbnf::k_max_gbnf_rules) {
    return false;
  }

  const emel::gbnf::grammar & grammar = *ev.request.grammar_out;
  uint64_t added_grammar_elements = 0;
  for (uint64_t i = 0; i < n_opt; ++i) {
    const uint32_t rec_rule_id = ctx.next_symbol_id + static_cast<uint32_t>(i);
    if (grammar.rule_lengths[rec_rule_id] != 0u) {
      return false;
    }
    const uint64_t rec_rule_len = prev_len + ((i > 0 || no_max) ? 1u : 0u) + 2u;
    if (rec_rule_len > emel::gbnf::k_max_gbnf_rule_elements) {
      return false;
    }
    added_grammar_elements += rec_rule_len;
  }

  if (grammar.element_count + added_grammar_elements > emel::gbnf::k_max_gbnf_elements) {
    return false;
  }

  const uint64_t final_rule_len = repeated_len + (n_opt > 0 ? 1u : 0u);
  return final_rule_len <= emel::gbnf::k_max_gbnf_rule_elements;
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

struct phase_ok {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none);
  }
};

struct phase_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct lexer_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct lexer_at_eof {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) && !ev.flow.has_token;
  }
};

struct lexer_has_token {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) && ev.flow.has_token;
  }
};

struct definition_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none);
  }
};

struct definition_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct expression_done_identifier {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.expression_kind == expression_parser::events::parse_kind::identifier;
  }
};

struct expression_done_non_identifier {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.expression_kind == expression_parser::events::parse_kind::non_identifier;
  }
};

struct expression_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct nonterm_definition_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.nonterm_mode == nonterm_parser::events::parse_mode::definition;
  }
};

struct nonterm_reference_done {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.nonterm_mode == nonterm_parser::events::parse_mode::reference;
  }
};

struct nonterm_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct term_from_need_term {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.current_term_origin == event::parse_flow::term_origin::need_term;
  }
};

struct term_from_after_term {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.current_term_origin == event::parse_flow::term_origin::after_term;
  }
};

struct term_failed {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

template <term_parser::events::term_kind kind>
struct term_kind_is {
  bool operator()(const event::parse_rules & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) && ev.flow.term_kind == kind;
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
    if (!literal_element_count(ev.flow.token.text, count)) {
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
    if (!character_class_element_count(ev.flow.token.text, count)) {
      return false;
    }
    return current_rule_has_space(ctx, count);
  }
};

struct token_rule_reference_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::rule_reference>{}(ev, ctx) &&
           current_rule_has_space(ctx, 1u) &&
           parse_rule_reference_text(ev.flow.token.text);
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

struct token_quantifier_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_kind_is<term_parser::events::term_kind::quantifier>{}(ev, ctx) &&
           is_quantifier_text(ev.flow.token.text) &&
           can_apply_quantifier(ev, ctx);
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

struct term_need_rule_reference_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_need_term{}(ev, ctx) && token_rule_reference_valid{}(ev, ctx);
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

struct term_after_rule_reference_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_rule_reference_valid{}(ev, ctx);
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

struct term_after_quantifier_valid {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return term_from_after_term{}(ev, ctx) && token_quantifier_valid{}(ev, ctx);
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
    return phase_ok{}(ev, ctx) && can_finalize_symbols(ev, ctx);
  }
};

struct eof_cannot_finalize_symbols {
  bool operator()(const event::parse_rules & ev, const action::context & ctx) const noexcept {
    return phase_ok{}(ev, ctx) && !can_finalize_symbols(ev, ctx);
  }
};

}  // namespace emel::gbnf::rule_parser::guard
