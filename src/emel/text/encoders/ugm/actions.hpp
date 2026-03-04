#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/ugm/context.hpp"
#include "emel/text/encoders/ugm/detail.hpp"

namespace emel::text::encoders::ugm::action {

namespace detail {

inline bool ugm_push_token(const event::encode & ev, const int32_t token, int32_t & count) noexcept {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t * base_ptrs[2] = {&sink, ev.token_ids.data()};
  int32_t * base = base_ptrs[static_cast<size_t>(has_buffer)];
  const bool non_negative_count = count >= 0;
  const int32_t safe_count = emel::text::encoders::ugm::detail::select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space = has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;
  const size_t target_index = count_index * static_cast<size_t>(write);
  int32_t * target = base + target_index;
  *target = emel::text::encoders::ugm::detail::select_i32(write, token, *target);
  count += static_cast<int32_t>(write);
  return write;
}

inline bool ugm_push_token_noop(const event::encode &, const int32_t, int32_t &) noexcept {
  return true;
}

inline bool ugm_push_token_if(const event::encode & ev,
                              const int32_t token,
                              int32_t & count,
                              const bool push_active) noexcept {
  using push_handler_t = bool (*)(const event::encode &, int32_t, int32_t &) noexcept;
  const push_handler_t push_handlers[2] = {
      ugm_push_token_noop,
      ugm_push_token,
  };
  return push_handlers[static_cast<size_t>(push_active)](ev, token, count);
}

inline int32_t lookup_token_exact(const emel::model::data::vocab & vocab,
                                  const std::string_view target) noexcept {
  int32_t resolved = emel::text::encoders::detail::k_token_null;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view token =
      emel::text::encoders::ugm::detail::ugm_token_text(vocab, static_cast<int32_t>(id));
    const bool exact = token == target;
    resolved =
      emel::text::encoders::ugm::detail::select_i32(exact, static_cast<int32_t>(id), resolved);
  }
  return resolved;
}

inline bool ugm_read_has_value_none(const emel::text::encoders::detail::naive_trie::node *) noexcept {
  return false;
}

inline bool ugm_read_has_value_some(const emel::text::encoders::detail::naive_trie::node * node) noexcept {
  return node->has_value;
}

inline int32_t ugm_read_token_none(const emel::text::encoders::detail::naive_trie::node *) noexcept {
  return 0;
}

inline int32_t ugm_read_token_some(const emel::text::encoders::detail::naive_trie::node * node) noexcept {
  return node->value;
}

inline const emel::text::encoders::detail::naive_trie::node * ugm_trie_step_none(
  const emel::text::encoders::detail::naive_trie::node *,
  const char) noexcept {
  return nullptr;
}

inline const emel::text::encoders::detail::naive_trie::node * ugm_trie_step_some(
  const emel::text::encoders::detail::naive_trie::node * node,
  const char c) noexcept {
  return emel::text::encoders::ugm::detail::ugm_trie_step(*node, c);
}

inline void run_dp_forward(const runtime::encode_runtime & ev, context & ctx) noexcept {
  const auto & vocab = *ctx.vocab;
  const std::string_view normalized = ev.normalized;
  const size_t safe_input_len = normalized.size();

  for (size_t input_offset = 0; input_offset < safe_input_len;) {
    const size_t n_utf8_code_units = std::min(
      static_cast<size_t>(emel::text::encoders::ugm::detail::ugm_utf8_len(normalized[input_offset])),
      safe_input_len - input_offset);
    bool single_codepoint_token_found = false;
    const auto current_best = ctx.best[input_offset];
    const auto *node = emel::text::encoders::ugm::detail::ugm_trie_root(
      ctx.token_matcher, normalized[input_offset]);
    using read_bool_handler_t = bool (*)(const emel::text::encoders::detail::naive_trie::node *) noexcept;
    const read_bool_handler_t read_has_value_handlers[2] = {
      ugm_read_has_value_none,
      ugm_read_has_value_some,
    };
    using read_i32_handler_t = int32_t (*)(const emel::text::encoders::detail::naive_trie::node *) noexcept;
    const read_i32_handler_t read_token_handlers[2] = {
      ugm_read_token_none,
      ugm_read_token_some,
    };
    using step_handler_t = const emel::text::encoders::detail::naive_trie::node * (*)(
      const emel::text::encoders::detail::naive_trie::node *,
      char) noexcept;
    const step_handler_t step_handlers[2] = {
      ugm_trie_step_none,
      ugm_trie_step_some,
    };

    const size_t max_prefix_steps = safe_input_len - input_offset;
    for (size_t step = 0; step < max_prefix_steps; ++step) {
      const size_t prefix_offset = input_offset + step + 1u;
      const bool active = node != nullptr;
      const bool has_value =
        read_has_value_handlers[static_cast<size_t>(active)](node);
      const bool single_codepoint = prefix_offset - input_offset == n_utf8_code_units;
      single_codepoint_token_found = single_codepoint_token_found || (has_value && single_codepoint);
      const int32_t token_id = read_token_handlers[static_cast<size_t>(active)](node);
      const bool token_id_valid = token_id >= 0 && static_cast<uint32_t>(token_id) < vocab.n_tokens;
      const uint32_t safe_token_id = emel::text::encoders::ugm::detail::select_u32(
        token_id_valid, static_cast<uint32_t>(token_id), 0u);
      const auto & token_data = vocab.entries[safe_token_id];
      const bool scored_value = has_value && token_id_valid;
      const bool is_user_defined = token_data.type == 4;
      const std::array<double, 2> score_table{
        static_cast<double>(token_data.score),
        0.0,
      };
      const double token_score = score_table[static_cast<size_t>(is_user_defined)];
      const double challenger_score = current_best.score_sum + token_score;
      auto & current_champ = ctx.best[prefix_offset];
      const bool better = scored_value && challenger_score > current_champ.score_sum;
      current_champ.token_id = emel::text::encoders::ugm::detail::select_i32(
        better, token_id, current_champ.token_id);
      current_champ.input_offset = emel::text::encoders::ugm::detail::select_u32(
        better, static_cast<uint32_t>(input_offset), current_champ.input_offset);
      current_champ.score_sum = emel::text::encoders::ugm::detail::select_f64(
        better, challenger_score, current_champ.score_sum);

      const bool can_advance = active && prefix_offset < safe_input_len;
      const size_t safe_offset =
        emel::text::encoders::ugm::detail::select_size(can_advance, prefix_offset, input_offset);
      const auto *next_node =
        step_handlers[static_cast<size_t>(active)](node, normalized[safe_offset]);
      const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
        node,
        next_node,
      };
      node = options[static_cast<size_t>(can_advance)];
    }

    const bool use_unk =
      !single_codepoint_token_found && ev.unk_id != emel::text::encoders::detail::k_token_null;
    const double challenger_score =
      current_best.score_sum + static_cast<double>(ctx.unknown_token_score);
    const size_t next_offset = input_offset + n_utf8_code_units;
    auto & current_champ = ctx.best[next_offset];
    const bool better = use_unk && challenger_score > current_champ.score_sum;
    current_champ.token_id =
      emel::text::encoders::ugm::detail::select_i32(better, ev.unk_id, current_champ.token_id);
    current_champ.input_offset = emel::text::encoders::ugm::detail::select_u32(
      better, static_cast<uint32_t>(input_offset), current_champ.input_offset);
    current_champ.score_sum = emel::text::encoders::ugm::detail::select_f64(
      better, challenger_score, current_champ.score_sum);

    input_offset += n_utf8_code_units;
  }

}

inline void run_dp_backtrace(const runtime::encode_runtime & ev, context & ctx) noexcept {
  const size_t safe_input_len = ev.normalized.size();
  size_t out_count = 0;
  bool is_prev_unknown = false;
  bool trace_failed = false;
  emel::text::encoders::ugm::action::best_tokenization tokenization = ctx.best[safe_input_len];
  bool trace_active = true;
  const size_t max_trace_steps = safe_input_len + 1u;
  for (size_t step = 0; step < max_trace_steps; ++step) {
    (void)step;
    const bool is_unknown = tokenization.token_id == ev.unk_id;
    const bool emit_token = trace_active && !(is_prev_unknown && is_unknown);
    const bool has_room = out_count < ctx.token_buffer.size();
    const bool write = emit_token && has_room;
    const size_t write_idx = out_count * static_cast<size_t>(write);
    ctx.token_buffer[write_idx] = emel::text::encoders::ugm::detail::select_i32(
      write, tokenization.token_id, ctx.token_buffer[write_idx]);
    out_count += static_cast<size_t>(write);
    trace_failed = trace_failed || (emit_token && !has_room);

    const bool at_root = tokenization.input_offset == 0u;
    const bool offset_valid = static_cast<size_t>(tokenization.input_offset) <= safe_input_len;
    const size_t next_index = emel::text::encoders::ugm::detail::select_size(
      offset_valid, static_cast<size_t>(tokenization.input_offset), safe_input_len);
    const auto next_tokenization = ctx.best[next_index];
    const bool advance = trace_active && !at_root && offset_valid;
    is_prev_unknown = emel::text::encoders::ugm::detail::select_bool(
      advance, is_unknown, is_prev_unknown);
    tokenization.token_id = emel::text::encoders::ugm::detail::select_i32(
      advance, next_tokenization.token_id, tokenization.token_id);
    tokenization.input_offset = emel::text::encoders::ugm::detail::select_u32(
      advance, next_tokenization.input_offset, tokenization.input_offset);
    tokenization.score_sum = emel::text::encoders::ugm::detail::select_f64(
      advance, next_tokenization.score_sum, tokenization.score_sum);
    const bool offset_invalid = trace_active && !offset_valid;
    trace_failed = trace_failed || offset_invalid;
    const bool trace_stop = trace_active && (at_root || offset_invalid);
    trace_active = trace_active && !trace_stop;
  }
  trace_failed = trace_failed || trace_active;
  ev.backtrace_failed = trace_failed;
  ev.traced_count = out_count * static_cast<size_t>(!trace_failed);
}

inline void emit_tokens(const runtime::encode_runtime & ev, context & ctx) noexcept {
  int32_t count = 0;
  bool emit_failed = false;
  const bool trace_count_valid = ev.traced_count <= ctx.token_buffer.size();
  emit_failed = emit_failed || !trace_count_valid;
  const size_t safe_traced_count = emel::text::encoders::ugm::detail::select_size(
    trace_count_valid, ev.traced_count, ctx.token_buffer.size());
  const size_t emit_limit = safe_traced_count;
  for (size_t i = 0; i < emit_limit; ++i) {
    const int32_t token = ctx.token_buffer[safe_traced_count - 1u - i];
    const bool push_active = !emit_failed;
    const bool pushed = ugm_push_token_if(ev.event_.request, token, count, push_active);
    emit_failed = emit_failed || (push_active && !pushed);
  }
  ev.emit_failed = emit_failed;
  ev.event_.ctx.token_count = count * static_cast<int32_t>(!emit_failed);
}

}  // namespace detail

struct begin_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.normalized = std::string_view{};
    ev.traced_count = 0u;
    ev.backtrace_failed = false;
    ev.emit_failed = false;
  }
};

struct begin_encode_sync_vocab {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    emel::text::encoders::action::sync_vocab(ev.event_, ctx);
    ctx.ugm_tables_ready = false;
    ctx.ugm_vocab = nullptr;
    ctx.token_matcher = emel::text::encoders::detail::naive_trie{};
    ctx.user_defined_token_matcher = emel::text::encoders::detail::naive_trie{};
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.normalized = std::string_view{};
    ev.traced_count = 0u;
    ev.backtrace_failed = false;
    ev.emit_failed = false;
  }
};

struct reject_invalid_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::reject_invalid_encode(ev.event_, ctx);
  }
};

struct resolve_vocab_unk {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.unk_id = ctx.vocab->unk_id;
  }
};

struct lookup_unk_id {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.unk_id = detail::lookup_token_exact(*ctx.vocab, "<unk>");
  }
};

struct normalize_input {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    std::string_view normalized{};
    const bool normalized_ok = emel::text::encoders::ugm::detail::normalize_ugm_into(
      *ctx.vocab, ctx, ev.event_.request.text, normalized);
    ev.normalized = normalized;
    ev.event_.ctx.err = emel::text::encoders::ugm::detail::select_i32(
      ev.event_.ctx.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok) && !normalized_ok,
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument),
      ev.event_.ctx.err);
  }
};

struct prepare_dp_input {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.traced_count = 0u;
    const size_t input_len = ev.normalized.size();
    const bool overflow = input_len >= ctx.best.size();
    ev.event_.ctx.err = emel::text::encoders::ugm::detail::select_i32(
      ev.event_.ctx.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok) && overflow,
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument),
      ev.event_.ctx.err);

    const bool setup_active = ev.event_.ctx.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok) && input_len > 0u;
    const size_t safe_input_len = input_len * static_cast<size_t>(setup_active);
    for (size_t i = 0; i <= safe_input_len; ++i) {
      ctx.best[i] = {ev.unk_id, 0u, -std::numeric_limits<double>::max()};
    }
    ctx.best[0] = {ev.unk_id, 0u, 0.0};
  }
};

struct run_dp_forward {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    detail::run_dp_forward(ev, ctx);
  }
};

struct run_dp_backtrace {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    detail::run_dp_backtrace(ev, ctx);
  }
};

struct run_dp_trace {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    detail::run_dp_forward(ev, ctx);
    detail::run_dp_backtrace(ev, ctx);
  }
};

struct emit_tokens {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    detail::emit_tokens(ev, ctx);
  }
};

struct mark_backtrace_failed {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.event_.ctx.token_count = 0;
    ev.event_.ctx.err = emel::text::encoders::error::to_emel(
      emel::text::encoders::error::code::invalid_argument);
  }
};

struct mark_emit_failed {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.event_.ctx.token_count = 0;
    ev.event_.ctx.err = emel::text::encoders::error::to_emel(
      emel::text::encoders::error::code::invalid_argument);
  }
};

struct sync_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *ctx.vocab);
    const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
    ev.event_.ctx.err = errors[static_cast<size_t>(ready)];
  }
};

struct mark_done {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::mark_done(ev.event_, ctx);
  }
};

struct ensure_last_error {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::ensure_last_error(ev.event_, ctx);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.token_count; ev.event_.ctx.err; }) {
      ev.event_.ctx.token_count = 0;
      ev.event_.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
    } else if constexpr (requires { ev.ctx.token_count; ev.ctx.err; }) {
      ev.ctx.token_count = 0;
      ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
    } else if constexpr (requires { ev.request; }) {
      emel::text::encoders::action::detail::signal_unexpected_request(ev.request);
    }
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr begin_encode_sync_vocab begin_encode_sync_vocab{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr resolve_vocab_unk resolve_vocab_unk{};
inline constexpr lookup_unk_id lookup_unk_id{};
inline constexpr normalize_input normalize_input{};
inline constexpr prepare_dp_input prepare_dp_input{};
inline constexpr run_dp_forward run_dp_forward{};
inline constexpr run_dp_backtrace run_dp_backtrace{};
inline constexpr run_dp_trace run_dp_trace{};
inline constexpr emit_tokens emit_tokens{};
inline constexpr mark_backtrace_failed mark_backtrace_failed{};
inline constexpr mark_emit_failed mark_emit_failed{};
inline constexpr sync_tables sync_tables{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::ugm::action
