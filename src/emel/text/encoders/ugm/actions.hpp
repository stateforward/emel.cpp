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

inline void run_dp_forward(const runtime::encode_runtime & ev, context & ctx) noexcept {
  const auto & vocab = *ctx.vocab;
  const std::string_view normalized = ev.normalized;
  const size_t safe_input_len = normalized.size();

  size_t input_offset = 0;
  while (input_offset < safe_input_len) {
    const size_t n_utf8_code_units = std::min(
      static_cast<size_t>(emel::text::encoders::ugm::detail::ugm_utf8_len(normalized[input_offset])),
      safe_input_len - input_offset);
    bool single_codepoint_token_found = false;
    const auto current_best = ctx.best[input_offset];
    size_t prefix_offset = input_offset;
    const auto *node = emel::text::encoders::ugm::detail::ugm_trie_root(
      ctx.token_matcher, normalized[prefix_offset]);
    prefix_offset += 1u;
    bool walking = node != nullptr && prefix_offset <= safe_input_len;

    while (walking) {
      const bool has_value = node->has_value;
      const bool single_codepoint = prefix_offset - input_offset == n_utf8_code_units;
      single_codepoint_token_found = single_codepoint_token_found || (has_value && single_codepoint);
      const int32_t token_id = node->value;
      const auto & token_data = vocab.entries[static_cast<uint32_t>(token_id)];
      const bool is_user_defined = token_data.type == 4;
      const std::array<double, 2> score_table{
        static_cast<double>(token_data.score),
        0.0,
      };
      const double token_score = score_table[static_cast<size_t>(is_user_defined)];
      const double challenger_score = current_best.score_sum + token_score;
      auto & current_champ = ctx.best[prefix_offset];
      const bool better = has_value && challenger_score > current_champ.score_sum;
      current_champ.token_id = emel::text::encoders::ugm::detail::select_i32(
        better, token_id, current_champ.token_id);
      current_champ.input_offset = emel::text::encoders::ugm::detail::select_u32(
        better, static_cast<uint32_t>(input_offset), current_champ.input_offset);
      current_champ.score_sum = emel::text::encoders::ugm::detail::select_f64(
        better, challenger_score, current_champ.score_sum);

      const bool can_advance = prefix_offset < safe_input_len;
      const size_t safe_offset =
        emel::text::encoders::ugm::detail::select_size(can_advance, prefix_offset, input_offset);
      const auto *next_node = emel::text::encoders::ugm::detail::ugm_trie_step(
        *node, normalized[safe_offset]);
      const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
        node,
        next_node,
      };
      node = options[static_cast<size_t>(can_advance)];
      prefix_offset += static_cast<size_t>(can_advance);
      walking = can_advance && node != nullptr && prefix_offset <= safe_input_len;
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
  bool tracing = true;
  while (tracing) {
    const bool is_unknown = tokenization.token_id == ev.unk_id;
    const bool emit_token = !(is_prev_unknown && is_unknown);
    const bool has_room = out_count < ctx.token_buffer.size();
    const bool write = !trace_failed && emit_token && has_room;
    const size_t write_idx = out_count * static_cast<size_t>(write);
    ctx.token_buffer[write_idx] = emel::text::encoders::ugm::detail::select_i32(
      write, tokenization.token_id, ctx.token_buffer[write_idx]);
    out_count += static_cast<size_t>(write);
    trace_failed = trace_failed || (!trace_failed && emit_token && !has_room);

    const bool at_root = tokenization.input_offset == 0u;
    const auto next_tokenization = ctx.best[tokenization.input_offset];
    const bool advance = !at_root;
    is_prev_unknown = emel::text::encoders::ugm::detail::select_bool(
      advance, is_unknown, is_prev_unknown);
    tokenization.token_id = emel::text::encoders::ugm::detail::select_i32(
      advance, next_tokenization.token_id, tokenization.token_id);
    tokenization.input_offset = emel::text::encoders::ugm::detail::select_u32(
      advance, next_tokenization.input_offset, tokenization.input_offset);
    tokenization.score_sum = emel::text::encoders::ugm::detail::select_f64(
      advance, next_tokenization.score_sum, tokenization.score_sum);
    tracing = !at_root;
  }

  ev.event_.ctx.err = emel::text::encoders::ugm::detail::select_i32(
      trace_failed,
      EMEL_ERR_INVALID_ARGUMENT,
      ev.event_.ctx.err);
  ev.traced_count = out_count * static_cast<size_t>(ev.event_.ctx.err == EMEL_OK);
}

inline void emit_tokens(const runtime::encode_runtime & ev, context & ctx) noexcept {
  int32_t count = 0;
  bool emit_failed = false;
  const size_t emit_limit = ev.traced_count;
  for (size_t i = 0; i < emit_limit; ++i) {
    const int32_t token = ctx.token_buffer[ev.traced_count - 1u - i];
    const bool push_active = !emit_failed;
    const bool pushed = ugm_push_token_if(ev.event_.request, token, count, push_active);
    emit_failed = emit_failed || (push_active && !pushed);
  }
  ev.event_.ctx.err = emel::text::encoders::ugm::detail::select_i32(
      emit_failed,
      EMEL_ERR_INVALID_ARGUMENT,
      ev.event_.ctx.err);
  ev.event_.ctx.token_count = count * static_cast<int32_t>(ev.event_.ctx.err == EMEL_OK);
}

}  // namespace detail

struct begin_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.normalized = std::string_view{};
    ev.traced_count = 0u;
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
      ev.event_.ctx.err == EMEL_OK && !normalized_ok,
      EMEL_ERR_INVALID_ARGUMENT,
      ev.event_.ctx.err);
  }
};

struct prepare_dp_input {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.traced_count = 0u;
    const size_t input_len = ev.normalized.size();
    const bool overflow = input_len >= ctx.best.size();
    ev.event_.ctx.err = emel::text::encoders::ugm::detail::select_i32(
      ev.event_.ctx.err == EMEL_OK && overflow,
      EMEL_ERR_INVALID_ARGUMENT,
      ev.event_.ctx.err);

    const bool setup_active = ev.event_.ctx.err == EMEL_OK && input_len > 0u;
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

struct sync_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::ugm::detail::ensure_ugm_tables(ctx, *ctx.vocab);
    const std::array<int32_t, 2> errors{EMEL_ERR_INVALID_ARGUMENT, EMEL_OK};
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
      ev.event_.ctx.err = EMEL_ERR_INVALID_ARGUMENT;
    } else if constexpr (requires { ev.ctx.token_count; ev.ctx.err; }) {
      ev.ctx.token_count = 0;
      ev.ctx.err = EMEL_ERR_INVALID_ARGUMENT;
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
inline constexpr sync_tables sync_tables{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::ugm::action
