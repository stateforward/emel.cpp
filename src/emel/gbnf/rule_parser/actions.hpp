#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

#include "emel/gbnf/rule_parser/expression_parser/events.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/detail.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/term_parser/events.hpp"

namespace emel::gbnf::rule_parser::action {

inline void append_unchecked(context & ctx, const emel::gbnf::element elem) noexcept {
  ctx.current_rule.elements[ctx.current_rule.size++] = elem;
}

inline void add_rule_unchecked(emel::gbnf::grammar & grammar,
                               const uint32_t rule_id,
                               const emel::gbnf::element * elements,
                               const uint32_t count) noexcept {
  grammar.rule_offsets[rule_id] = grammar.element_count;
  grammar.rule_lengths[rule_id] = count;
  std::memcpy(grammar.elements.data() + grammar.element_count,
              elements,
              sizeof(emel::gbnf::element) * count);
  grammar.element_count += count;
  if (rule_id + 1u > grammar.rule_count) {
    grammar.rule_count = rule_id + 1u;
  }
}

inline bool on_lexer_done(void * owner, const lexer::events::next_done & ev) noexcept {
  auto * ctx = static_cast<event::parse_rules_ctx *>(owner);
  ctx->err = emel::error::cast(error::none);
  ctx->has_token = ev.has_token;
  ctx->token = ev.token;
  ctx->cursor = ev.next_cursor;
  ctx->expression_kind = expression_parser::events::parse_kind::unknown;
  ctx->term_kind = term_parser::events::term_kind::unknown;
  return true;
}

inline bool on_lexer_error(void * owner, const lexer::events::next_error & ev) noexcept {
  auto * ctx = static_cast<event::parse_rules_ctx *>(owner);
  ctx->err = static_cast<emel::error::type>(ev.err);
  ctx->has_token = false;
  ctx->token = {};
  ctx->expression_kind = expression_parser::events::parse_kind::unknown;
  ctx->term_kind = term_parser::events::term_kind::unknown;
  return true;
}

struct reject_invalid_parse_with_dispatch {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.request.grammar_out->reset();
    ev.request.dispatch_error(events::parsing_error{
      *ev.request.grammar_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_parse_with_grammar_only {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.request.grammar_out->reset();
  }
};

struct reject_invalid_parse_without_grammar {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_parse {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.request.grammar_out->reset();
    ev.ctx.cursor = lexer::cursor{
      .input = ev.request.grammar_text,
      .offset = 0,
      .token_count = 0,
    };
    ev.ctx.token = {};
    ev.ctx.has_token = false;
    ev.ctx.nonterm_mode = nonterm_parser::events::parse_mode::none;
    ev.ctx.nonterm_rule_id = 0;
    ev.ctx.expression_kind = expression_parser::events::parse_kind::unknown;
    ev.ctx.term_kind = term_parser::events::term_kind::unknown;
    ev.ctx.current_term_origin = event::parse_rules_ctx::term_origin::none;

    ctx.current_rule_id = 0;
    ctx.current_rule.size = 0;
    ctx.last_sym_start = 0;
    ctx.group_depth = 0;
    ctx.symbols.clear();
    ctx.rule_defined.fill(false);
    ctx.next_symbol_id = 0;
  }
};

struct request_next_token {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::internal_error);
    ev.ctx.has_token = false;
    ev.ctx.token = {};
    ev.ctx.nonterm_mode = nonterm_parser::events::parse_mode::none;
    ev.ctx.nonterm_rule_id = 0;
    ev.ctx.expression_kind = expression_parser::events::parse_kind::unknown;
    ev.ctx.term_kind = term_parser::events::term_kind::unknown;

    const callback<bool(const lexer::events::next_done &)> done_cb{&ev.ctx, on_lexer_done};
    const callback<bool(const lexer::events::next_error &)> error_cb{&ev.ctx, on_lexer_error};
    const lexer::event::next next_ev{
      ev.ctx.cursor,
      done_cb,
      error_cb,
    };
    (void)ctx.lexer.process_event(next_ev);
  }
};

struct consume_token_invalid {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::parse_failed);
  }
};

struct fail_eof_in_expect_definition {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::parse_failed);
  }
};

struct set_nonterm_mode_definition {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.nonterm_mode = nonterm_parser::events::parse_mode::definition;
  }
};

struct set_nonterm_mode_reference {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.nonterm_mode = nonterm_parser::events::parse_mode::reference;
  }
};

struct set_term_origin_need_term {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.current_term_origin = event::parse_rules_ctx::term_origin::need_term;
  }
};

struct set_term_origin_after_term {
  void operator()(const event::parse_rules & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.current_term_origin = event::parse_rules_ctx::term_origin::after_term;
  }
};

struct apply_nonterm_definition {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ctx.current_rule_id = ev.ctx.nonterm_rule_id;
  }
};

struct apply_nonterm_reference {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ctx.last_sym_start = ctx.current_rule.size;
    append_unchecked(ctx, {emel::gbnf::element_type::rule_ref, ev.ctx.nonterm_rule_id});
  }
};

struct consume_token_definition_operator {
  void operator()(context & ctx) const noexcept {
    ctx.current_rule.size = 0;
    ctx.last_sym_start = 0;
    ctx.group_depth = 0;
  }
};

struct consume_token_alternation {
  void operator()(context & ctx) const noexcept {
    append_unchecked(ctx, {emel::gbnf::element_type::alt, 0});
  }
};

struct consume_token_literal {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ctx.last_sym_start = ctx.current_rule.size;
    const std::string_view text = ev.ctx.token.text;
    const char * pos = text.data() + 1u;
    const char * end = text.data() + text.size() - 1u;
    while (pos < end) {
      const auto parsed = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      append_unchecked(ctx, {emel::gbnf::element_type::character, parsed.first});
      pos = parsed.second;
    }
  }
};

struct consume_token_character_class {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    ctx.last_sym_start = ctx.current_rule.size;
    const std::string_view text = ev.ctx.token.text;
    const char * pos = text.data() + 1u;
    const char * end = text.data() + text.size() - 1u;
    emel::gbnf::element_type start_type = emel::gbnf::element_type::character;
    if (pos < end && *pos == '^') {
      start_type = emel::gbnf::element_type::char_not;
      ++pos;
    }

    bool first = true;
    while (pos < end) {
      const auto first_char = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      const auto lead_type = first ? start_type : emel::gbnf::element_type::char_alt;
      append_unchecked(ctx, {lead_type, first_char.first});
      first = false;
      pos = first_char.second;

      if (pos + 1u < end && pos[0] == '-' && pos[1] != ']') {
        ++pos;
        const auto range_char = emel::gbnf::rule_parser::detail::parse_char(pos, end);
        append_unchecked(ctx, {emel::gbnf::element_type::char_rng_upper, range_char.first});
        pos = range_char.second;
      }
    }
  }
};

struct consume_token_rule_reference {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    bool token_not = false;
    uint32_t token_id = 0;
    const std::string_view text = ev.ctx.token.text;
    std::size_t pos = 0;
    if (text[0] == '!') {
      token_not = true;
      pos = 1u;
    }
    pos += 2u;

    uint64_t value = 0;
    const char * cursor = text.data() + pos;
    const char * end = text.data() + text.size();
    const char * next = nullptr;
    (void)emel::gbnf::rule_parser::detail::parse_uint64(cursor, end, value, &next);
    token_id = static_cast<uint32_t>(value);

    const auto type = token_not ? emel::gbnf::element_type::token_not
                                : emel::gbnf::element_type::token;
    ctx.last_sym_start = ctx.current_rule.size;
    append_unchecked(ctx, {type, token_id});
  }
};

struct finalize_active_rule_on_eof {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    append_unchecked(ctx, {emel::gbnf::element_type::end, 0});
    add_rule_unchecked(*ev.request.grammar_out,
                       ctx.current_rule_id,
                       ctx.current_rule.elements.data(),
                       ctx.current_rule.size);
    ctx.current_rule.size = 0;
    ctx.last_sym_start = 0;
  }
};

struct consume_token_dot {
  void operator()(context & ctx) const noexcept {
    ctx.last_sym_start = ctx.current_rule.size;
    append_unchecked(ctx, {emel::gbnf::element_type::char_any, 0});
  }
};

struct consume_token_open_group {
  void operator()(context & ctx) const noexcept {
    const uint32_t generated_rule_id = ctx.next_symbol_id++;
    ctx.rule_defined[generated_rule_id] = true;
    ctx.group_stack[ctx.group_depth] = context::group_frame{
      .sequence_start = ctx.current_rule.size,
      .generated_rule_id = generated_rule_id,
    };
    ++ctx.group_depth;
  }
};

struct consume_token_close_group {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    const context::group_frame frame = ctx.group_stack[ctx.group_depth - 1u];
    const uint32_t group_len = ctx.current_rule.size - frame.sequence_start;
    emel::gbnf::element * const tmp = ctx.group_scratch.get();
    std::memcpy(tmp,
                ctx.current_rule.elements.data() + frame.sequence_start,
                sizeof(emel::gbnf::element) * group_len);
    tmp[group_len] = {emel::gbnf::element_type::end, 0};
    add_rule_unchecked(*ev.request.grammar_out, frame.generated_rule_id, tmp, group_len + 1u);

    ctx.current_rule.size = frame.sequence_start;
    ctx.last_sym_start = ctx.current_rule.size;
    append_unchecked(ctx, {emel::gbnf::element_type::rule_ref, frame.generated_rule_id});
    --ctx.group_depth;
  }
};

struct consume_token_quantifier {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();

    uint64_t min_times = 0;
    uint64_t max_times = 0;
    const std::string_view text = ev.ctx.token.text;

    if (text == "*") {
      min_times = 0;
      max_times = k_no_max;
    } else if (text == "+") {
      min_times = 1;
      max_times = k_no_max;
    } else if (text == "?") {
      min_times = 0;
      max_times = 1;
    } else {
      const char * cursor = text.data() + 1u;
      const char * end = text.data() + text.size() - 1u;
      const char * next = nullptr;
      (void)emel::gbnf::rule_parser::detail::parse_uint64(cursor, end, min_times, &next);
      if (next == end) {
        max_times = min_times;
      } else if (next + 1u == end) {
        max_times = k_no_max;
      } else {
        ++next;
        (void)emel::gbnf::rule_parser::detail::parse_uint64(next, end, max_times, &next);
      }
    }

    const uint32_t prev_len = ctx.current_rule.size - ctx.last_sym_start;
    emel::gbnf::element * const prev_elements = ctx.prev_scratch.get();
    std::memcpy(prev_elements,
                ctx.current_rule.elements.data() + ctx.last_sym_start,
                sizeof(emel::gbnf::element) * prev_len);

    if (min_times == 0) {
      ctx.current_rule.size = ctx.last_sym_start;
    } else {
      for (uint64_t i = 1; i < min_times; ++i) {
        std::memcpy(ctx.current_rule.elements.data() + ctx.current_rule.size,
                    prev_elements,
                    sizeof(emel::gbnf::element) * prev_len);
        ctx.current_rule.size += prev_len;
      }
    }

    const bool no_max = max_times == k_no_max;
    const uint64_t n_opt = no_max ? 1 : (max_times - min_times);
    uint32_t last_rec_rule_id = 0;
    emel::gbnf::element * const rec_elements = ctx.rec_scratch.get();

    for (uint64_t i = 0; i < n_opt; ++i) {
      uint32_t rec_len = 0;
      std::memcpy(rec_elements, prev_elements, sizeof(emel::gbnf::element) * prev_len);
      rec_len += prev_len;

      const uint32_t rec_rule_id = ctx.next_symbol_id++;
      ctx.rule_defined[rec_rule_id] = true;
      if (i > 0 || no_max) {
        const uint32_t ref_id = no_max ? rec_rule_id : last_rec_rule_id;
        rec_elements[rec_len++] = {emel::gbnf::element_type::rule_ref, ref_id};
      }
      rec_elements[rec_len++] = {emel::gbnf::element_type::alt, 0};
      rec_elements[rec_len++] = {emel::gbnf::element_type::end, 0};
      add_rule_unchecked(*ev.request.grammar_out, rec_rule_id, rec_elements, rec_len);
      last_rec_rule_id = rec_rule_id;
    }

    if (n_opt > 0) {
      append_unchecked(ctx, {emel::gbnf::element_type::rule_ref, last_rec_rule_id});
    }
  }
};

struct dispatch_done {
  void operator()(const event::parse_rules & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::parsing_done{*ev.request.grammar_out});
  }
};

struct dispatch_error {
  void operator()(const event::parse_rules & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::parsing_error{
      *ev.request.grammar_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_parse_with_dispatch reject_invalid_parse_with_dispatch{};
inline constexpr reject_invalid_parse_with_grammar_only reject_invalid_parse_with_grammar_only{};
inline constexpr reject_invalid_parse_without_grammar reject_invalid_parse_without_grammar{};
inline constexpr begin_parse begin_parse{};
inline constexpr request_next_token request_next_token{};
inline constexpr consume_token_invalid consume_token_invalid{};
inline constexpr fail_eof_in_expect_definition fail_eof_in_expect_definition{};
inline constexpr set_nonterm_mode_definition set_nonterm_mode_definition{};
inline constexpr set_nonterm_mode_reference set_nonterm_mode_reference{};
inline constexpr set_term_origin_need_term set_term_origin_need_term{};
inline constexpr set_term_origin_after_term set_term_origin_after_term{};
inline constexpr apply_nonterm_definition apply_nonterm_definition{};
inline constexpr apply_nonterm_reference apply_nonterm_reference{};
inline constexpr consume_token_definition_operator consume_token_definition_operator{};
inline constexpr consume_token_alternation consume_token_alternation{};
inline constexpr consume_token_literal consume_token_literal{};
inline constexpr consume_token_character_class consume_token_character_class{};
inline constexpr consume_token_rule_reference consume_token_rule_reference{};
inline constexpr finalize_active_rule_on_eof finalize_active_rule_on_eof{};
inline constexpr consume_token_dot consume_token_dot{};
inline constexpr consume_token_open_group consume_token_open_group{};
inline constexpr consume_token_close_group consume_token_close_group{};
inline constexpr consume_token_quantifier consume_token_quantifier{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::action
