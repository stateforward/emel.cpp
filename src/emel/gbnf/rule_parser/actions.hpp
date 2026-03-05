#pragma once

#include <algorithm>
#include <array>
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
  grammar.rule_count = std::max(grammar.rule_count, rule_id + 1u);
}

inline bool can_apply_quantifier_bounds(const event::parse_rules & ev,
                                        const context & ctx,
                                        const uint64_t min_times,
                                        const uint64_t max_times) noexcept {
  constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t k_max_repetition_threshold = 2000;
  if (ctx.last_sym_start == ctx.current_rule.size) {
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

inline emel::gbnf::element_type select_char_class_lead_type(
    const bool first,
    const emel::gbnf::element_type start_type) noexcept {
  constexpr std::array<emel::gbnf::element_type, 2> lead_types = {
      emel::gbnf::element_type::char_alt,
      emel::gbnf::element_type::char_alt,
  };
  std::array<emel::gbnf::element_type, 2> lead_type_candidates = lead_types;
  lead_type_candidates[1] = start_type;
  return lead_type_candidates[static_cast<size_t>(first)];
}

inline bool parse_rule_reference_digits_text(const std::string_view text,
                                             uint32_t & token_id) noexcept {
  static constexpr char k_zero = '\0';
  const bool has_data = text.data() != nullptr;
  const uintptr_t data_addr = emel::gbnf::rule_parser::detail::select_uptr(
      has_data, reinterpret_cast<uintptr_t>(text.data()),
      reinterpret_cast<uintptr_t>(&k_zero));
  const char * safe_data = reinterpret_cast<const char *>(data_addr);

  uint64_t value = 0;
  const char * cursor = safe_data;
  const char * end = safe_data + text.size();
  const char * next = cursor;
  const bool parsed_uint =
      emel::gbnf::rule_parser::detail::parse_uint64(cursor, end, value, &next);
  const std::size_t pos = static_cast<std::size_t>(next - safe_data);
  const bool value_in_range = value <= std::numeric_limits<uint32_t>::max();
  const bool consumed_all = text.size() == pos;
  const bool valid = parsed_uint && value_in_range && consumed_all;
  token_id = emel::gbnf::rule_parser::detail::select_u32(
      valid, static_cast<uint32_t>(value), token_id);
  return valid;
}

inline void append_optional_rule_ref_none(context &, const uint32_t) noexcept {}

inline void append_optional_rule_ref_some(context & ctx, const uint32_t rule_id) noexcept {
  append_unchecked(ctx, {emel::gbnf::element_type::rule_ref, rule_id});
}

inline void apply_quantifier_bounds(const event::parse_rules & ev,
                                    context & ctx,
                                    const uint64_t min_times,
                                    const uint64_t max_times) noexcept {
  constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();

  const uint32_t prev_len = ctx.current_rule.size - ctx.last_sym_start;
  emel::gbnf::element * const prev_elements = ctx.prev_scratch.get();
  std::memcpy(prev_elements,
              ctx.current_rule.elements.data() + ctx.last_sym_start,
              sizeof(emel::gbnf::element) * prev_len);

  for (uint64_t i = 1; i < min_times; ++i) {
    std::memcpy(ctx.current_rule.elements.data() + ctx.current_rule.size,
                prev_elements,
                sizeof(emel::gbnf::element) * prev_len);
    ctx.current_rule.size += prev_len;
  }
  const std::array<uint32_t, 2> rule_sizes = {ctx.current_rule.size, ctx.last_sym_start};
  ctx.current_rule.size = rule_sizes[static_cast<size_t>(min_times == 0)];

  const bool no_max = max_times == k_no_max;
  const std::array<uint64_t, 2> n_opt_candidates = {max_times - min_times, 1u};
  const uint64_t n_opt = n_opt_candidates[static_cast<size_t>(no_max)];
  uint32_t last_rec_rule_id = 0;
  emel::gbnf::element * const rec_elements = ctx.rec_scratch.get();

  for (uint64_t i = 0; i < n_opt; ++i) {
    uint32_t rec_len = 0;
    std::memcpy(rec_elements, prev_elements, sizeof(emel::gbnf::element) * prev_len);
    rec_len += prev_len;

    const uint32_t rec_rule_id = ctx.next_symbol_id++;
    ctx.rule_defined[rec_rule_id] = true;
    const size_t append_ref = static_cast<size_t>(i > 0 || no_max);
    const std::array<uint32_t, 2> ref_id_candidates = {last_rec_rule_id, rec_rule_id};
    const uint32_t ref_id = ref_id_candidates[static_cast<size_t>(no_max)];
    rec_elements[rec_len] = {emel::gbnf::element_type::rule_ref, ref_id};
    rec_len += static_cast<uint32_t>(append_ref);
    rec_elements[rec_len++] = {emel::gbnf::element_type::alt, 0};
    rec_elements[rec_len++] = {emel::gbnf::element_type::end, 0};
    add_rule_unchecked(*ev.request.grammar_out, rec_rule_id, rec_elements, rec_len);
    last_rec_rule_id = rec_rule_id;
  }

  constexpr std::array<void (*)(context &, uint32_t), 2> optional_rule_ref_handlers = {
      append_optional_rule_ref_none,
      append_optional_rule_ref_some,
  };
  optional_rule_ref_handlers[static_cast<size_t>(n_opt > 0)](ctx, last_rec_rule_id);
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
    ev.ctx.nonterm_lookup_hash = 0;
    ev.ctx.nonterm_lookup_rule_id = 0;
    ev.ctx.nonterm_lookup_found = false;
    ev.ctx.nonterm_lookup_can_insert = false;
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
    ev.ctx.nonterm_lookup_hash = 0;
    ev.ctx.nonterm_lookup_rule_id = 0;
    ev.ctx.nonterm_lookup_found = false;
    ev.ctx.nonterm_lookup_can_insert = false;
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
    const std::string_view text = ev.ctx.token.text;
    static constexpr char k_zero = '\0';
    const bool has_data = text.data() != nullptr;
    const uintptr_t data_addr = emel::gbnf::rule_parser::detail::select_uptr(
        has_data,
        reinterpret_cast<uintptr_t>(text.data()),
        reinterpret_cast<uintptr_t>(&k_zero));
    const char * const safe_data = reinterpret_cast<const char *>(data_addr);
    const bool has_envelope = text.size() >= 2u;
    const size_t start_offset =
        emel::gbnf::rule_parser::detail::select_size(has_envelope, 1u, 0u);
    const size_t end_offset =
        emel::gbnf::rule_parser::detail::select_size(has_envelope, text.size() - 1u, 0u);

    const uint32_t original_size = ctx.current_rule.size;
    ctx.last_sym_start = original_size;
    const char * pos = safe_data + start_offset;
    const char * end = safe_data + end_offset;
    bool ok = has_envelope;
    while (ok && pos < end) {
      const auto parsed = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      ok = parsed.second != nullptr;
      if (!ok) {
        break;
      }
      append_unchecked(ctx, {emel::gbnf::element_type::character, parsed.first});
      pos = parsed.second;
    }

    const std::array<uint32_t, 2> size_candidates = {
        original_size,
        ctx.current_rule.size,
    };
    ctx.current_rule.size = size_candidates[static_cast<size_t>(ok)];
    const std::array<emel::error::type, 2> error_candidates = {
        emel::error::cast(error::parse_failed),
        emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_character_class {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    const std::string_view text = ev.ctx.token.text;
    static constexpr char k_zero = '\0';
    const bool has_data = text.data() != nullptr;
    const uintptr_t data_addr = emel::gbnf::rule_parser::detail::select_uptr(
        has_data,
        reinterpret_cast<uintptr_t>(text.data()),
        reinterpret_cast<uintptr_t>(&k_zero));
    const char * const safe_data = reinterpret_cast<const char *>(data_addr);
    const bool has_envelope = text.size() >= 2u;
    const size_t start_offset =
        emel::gbnf::rule_parser::detail::select_size(has_envelope, 1u, 0u);
    const size_t end_offset =
        emel::gbnf::rule_parser::detail::select_size(has_envelope, text.size() - 1u, 0u);

    const uint32_t original_size = ctx.current_rule.size;
    ctx.last_sym_start = original_size;
    const char * pos = safe_data + start_offset;
    const char * end = safe_data + end_offset;
    const bool leading_not = pos < end && *pos == '^';
    const emel::gbnf::element_type start_types[2] = {
        emel::gbnf::element_type::character,
        emel::gbnf::element_type::char_not};
    const emel::gbnf::element_type start_type =
        start_types[static_cast<size_t>(leading_not)];
    pos += static_cast<size_t>(leading_not);
    bool first = true;
    bool ok = has_envelope;
    while (ok && pos < end) {
      const auto parsed = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      ok = parsed.second != nullptr;
      if (!ok) {
        break;
      }
      append_unchecked(ctx, {select_char_class_lead_type(first, start_type), parsed.first});
      first = false;
      pos = parsed.second;

      const bool has_range = pos + 1u < end && pos[0] == '-' && pos[1] != ']';
      if (!has_range) {
        continue;
      }

      ++pos;
      const auto range_char = emel::gbnf::rule_parser::detail::parse_char(pos, end);
      ok = range_char.second != nullptr;
      if (!ok) {
        break;
      }
      append_unchecked(ctx, {emel::gbnf::element_type::char_rng_upper, range_char.first});
      pos = range_char.second;
    }

    const std::array<uint32_t, 2> size_candidates = {
        original_size,
        ctx.current_rule.size,
    };
    ctx.current_rule.size = size_candidates[static_cast<size_t>(ok)];
    const std::array<emel::error::type, 2> error_candidates = {
        emel::error::cast(error::parse_failed),
        emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

inline void append_rule_reference_plain_none(context &, const uint32_t) noexcept {}

inline void append_rule_reference_plain_some(context & ctx, const uint32_t token_id) noexcept {
  ctx.last_sym_start = ctx.current_rule.size;
  append_unchecked(ctx, {emel::gbnf::element_type::token, token_id});
}

inline void append_rule_reference_negated_none(context &, const uint32_t) noexcept {}

inline void append_rule_reference_negated_some(context & ctx, const uint32_t token_id) noexcept {
  ctx.last_sym_start = ctx.current_rule.size;
  append_unchecked(ctx, {emel::gbnf::element_type::token_not, token_id});
}

struct consume_token_rule_reference_plain {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    uint32_t token_id = 0;
    const std::string_view text = ev.ctx.token.text;
    const std::string_view digits = text.substr(2u, text.size() - 4u);
    const bool parsed = parse_rule_reference_digits_text(digits, token_id);
    constexpr std::array<void (*)(context &, uint32_t), 2> append_handlers = {
      append_rule_reference_plain_none,
      append_rule_reference_plain_some,
    };
    append_handlers[static_cast<size_t>(parsed)](ctx, token_id);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(parsed)];
  }
};

struct consume_token_rule_reference_negated {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    uint32_t token_id = 0;
    const std::string_view text = ev.ctx.token.text;
    const std::string_view digits = text.substr(3u, text.size() - 5u);
    const bool parsed = parse_rule_reference_digits_text(digits, token_id);
    constexpr std::array<void (*)(context &, uint32_t), 2> append_handlers = {
      append_rule_reference_negated_none,
      append_rule_reference_negated_some,
    };
    append_handlers[static_cast<size_t>(parsed)](ctx, token_id);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(parsed)];
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

inline void apply_quantifier_bounds_none(const event::parse_rules &,
                                         context &,
                                         const uint64_t,
                                         const uint64_t) noexcept {}

inline void apply_quantifier_bounds_some(const event::parse_rules & ev,
                                         context & ctx,
                                         const uint64_t min_times,
                                         const uint64_t max_times) noexcept {
  apply_quantifier_bounds(ev, ctx, min_times, max_times);
}

struct consume_token_quantifier_star {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok = can_apply_quantifier_bounds(ev, ctx, 0u, k_no_max);
    handlers[static_cast<size_t>(ok)](ev, ctx, 0u, k_no_max);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_quantifier_plus {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok = can_apply_quantifier_bounds(ev, ctx, 1u, k_no_max);
    handlers[static_cast<size_t>(ok)](ev, ctx, 1u, k_no_max);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_quantifier_question {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok = can_apply_quantifier_bounds(ev, ctx, 0u, 1u);
    handlers[static_cast<size_t>(ok)](ev, ctx, 0u, 1u);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_quantifier_braced_exact {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    const std::string_view text = ev.ctx.token.text;
    const std::string_view digits = text.substr(1u, text.size() - 2u);
    uint32_t parsed_min = 0;
    const bool parsed = parse_rule_reference_digits_text(digits, parsed_min);
    const uint64_t min_times = static_cast<uint64_t>(parsed_min);
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok = parsed && can_apply_quantifier_bounds(ev, ctx, min_times, min_times);
    handlers[static_cast<size_t>(ok)](ev, ctx, min_times, min_times);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_quantifier_braced_open {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    constexpr uint64_t k_no_max = std::numeric_limits<uint64_t>::max();
    const std::string_view text = ev.ctx.token.text;
    const std::string_view digits = text.substr(1u, text.size() - 3u);
    uint32_t parsed_min = 0;
    const bool parsed = parse_rule_reference_digits_text(digits, parsed_min);
    const uint64_t min_times = static_cast<uint64_t>(parsed_min);
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok = parsed && can_apply_quantifier_bounds(ev, ctx, min_times, k_no_max);
    handlers[static_cast<size_t>(ok)](ev, ctx, min_times, k_no_max);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
  }
};

struct consume_token_quantifier_braced_range {
  void operator()(const event::parse_rules & ev, context & ctx) const noexcept {
    const std::string_view text = ev.ctx.token.text;
    const std::string_view core = text.substr(1u, text.size() - 2u);
    const size_t comma_pos = core.find(',');
    const size_t has_comma = static_cast<size_t>(comma_pos != std::string_view::npos);
    const size_t safe_comma_pos =
        emel::gbnf::rule_parser::detail::select_size(has_comma != 0u, comma_pos, 0u);
    const std::string_view min_digits = core.substr(0u, safe_comma_pos);
    const size_t max_offset = safe_comma_pos + has_comma;
    const std::string_view max_digits = core.substr(max_offset, core.size() - max_offset);

    uint32_t parsed_min = 0;
    uint32_t parsed_max = 0;
    const bool min_ok = parse_rule_reference_digits_text(min_digits, parsed_min);
    const bool max_ok = parse_rule_reference_digits_text(max_digits, parsed_max);
    const uint64_t min_times = static_cast<uint64_t>(parsed_min);
    const uint64_t max_times = static_cast<uint64_t>(parsed_max);
    constexpr std::array<void (*)(const event::parse_rules &, context &, uint64_t, uint64_t), 2>
        handlers = {
          apply_quantifier_bounds_none,
          apply_quantifier_bounds_some,
        };
    const bool ok =
        min_ok && max_ok && can_apply_quantifier_bounds(ev, ctx, min_times, max_times);
    handlers[static_cast<size_t>(ok)](ev, ctx, min_times, max_times);
    const std::array<emel::error::type, 2> error_candidates = {
      emel::error::cast(error::parse_failed),
      emel::error::cast(error::none),
    };
    ev.ctx.err = error_candidates[static_cast<size_t>(ok)];
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
inline constexpr consume_token_rule_reference_plain consume_token_rule_reference_plain{};
inline constexpr consume_token_rule_reference_negated consume_token_rule_reference_negated{};
inline constexpr finalize_active_rule_on_eof finalize_active_rule_on_eof{};
inline constexpr consume_token_dot consume_token_dot{};
inline constexpr consume_token_open_group consume_token_open_group{};
inline constexpr consume_token_close_group consume_token_close_group{};
inline constexpr consume_token_quantifier_star consume_token_quantifier_star{};
inline constexpr consume_token_quantifier_plus consume_token_quantifier_plus{};
inline constexpr consume_token_quantifier_question consume_token_quantifier_question{};
inline constexpr consume_token_quantifier_braced_exact consume_token_quantifier_braced_exact{};
inline constexpr consume_token_quantifier_braced_open consume_token_quantifier_braced_open{};
inline constexpr consume_token_quantifier_braced_range consume_token_quantifier_braced_range{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::action
