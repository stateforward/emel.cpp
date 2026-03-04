#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/rwkv/context.hpp"
#include "emel/text/encoders/rwkv/detail.hpp"

namespace emel::text::encoders::rwkv::action {

namespace detail {

struct unk_lookup_result {
  int32_t id = emel::text::encoders::detail::k_token_null;
  bool found = false;
};

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

template <class pointer_type>
inline pointer_type * select_ptr(const bool choose_true,
                                 pointer_type * true_value,
                                 pointer_type * false_value) noexcept {
  const uintptr_t mask = static_cast<uintptr_t>(0) - static_cast<uintptr_t>(choose_true);
  const uintptr_t t = reinterpret_cast<uintptr_t>(true_value);
  const uintptr_t f = reinterpret_cast<uintptr_t>(false_value);
  return reinterpret_cast<pointer_type *>((f & ~mask) | (t & mask));
}

inline bool rwkv_push_token(const event::encode & ev,
                            const int32_t token,
                            int32_t & count) noexcept {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t * base_ptrs[2] = {&sink, ev.token_ids.data()};
  int32_t * base = base_ptrs[static_cast<size_t>(has_buffer)];
  const bool non_negative_count = count >= 0;
  const int32_t safe_count =
    emel::text::encoders::rwkv::detail::select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space = has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;
  const size_t target_index = count_index * static_cast<size_t>(write);
  int32_t * target = base + target_index;
  *target = emel::text::encoders::rwkv::detail::select_i32(write, token, *target);
  count += static_cast<int32_t>(write);
  return write;
}

inline unk_lookup_result lookup_unk_candidate(const emel::model::data::vocab & vocab) {
  auto process_text_none = +[](const std::string_view,
                               const int32_t,
                               const std::string_view,
                               std::string &,
                               int32_t &,
                               bool &) noexcept {};
  auto process_text_some = +[](const std::string_view text_value,
                               const int32_t id_value,
                               const std::string_view target_value,
                               std::string & unescaped_value,
                               int32_t & resolved_value,
                               bool & done_value) noexcept {
    const bool ok = emel::text::encoders::rwkv::detail::unescape_rwkv_token(
      text_value, unescaped_value);
    const bool match = ok && unescaped_value == target_value;
    resolved_value = emel::text::encoders::rwkv::detail::select_i32(
      match, id_value, resolved_value);
    done_value = done_value || match;
  };

  int32_t resolved = emel::text::encoders::detail::k_token_null;
  std::string unescaped;
  bool loop_active = true;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const bool step_active = loop_active;
    const std::string_view text =
      emel::text::encoders::rwkv::detail::rwkv_token_text(vocab, static_cast<int32_t>(id));
    using process_text_handler_t = void (*)(std::string_view,
                                            int32_t,
                                            std::string_view,
                                            std::string &,
                                            int32_t &,
                                            bool &);
    const process_text_handler_t process_text_handlers[2] = {
      process_text_none,
      process_text_some,
    };
    bool step_done = false;
    process_text_handlers[static_cast<size_t>(step_active && !text.empty())](
      text, static_cast<int32_t>(id), "<unk>", unescaped, resolved, step_done);
    loop_active = loop_active && !step_done;
  }

  return unk_lookup_result{
    .id = resolved,
    .found = resolved != emel::text::encoders::detail::k_token_null,
  };
}

inline void run_encode_tokens(const runtime::encode_runtime & ev, context & ctx) noexcept {
  int32_t count = 0;
  size_t position = 0;
  bool active = ev.event_.ctx.err == EMEL_OK;
  const std::string_view text = ev.event_.request.text;

  while (active && position < text.size()) {
    const auto * node = ctx.token_matcher.traverse(text[position]);
    int32_t token_id = ev.unk_id;
    size_t token_end = position + 1u;
    size_t offset = position + 1u;
    const auto * walk = node;

    while (walk != nullptr) {
      token_id = emel::text::encoders::rwkv::detail::select_i32(
        walk->has_value, walk->value, token_id);
      token_end = select_size(
        walk->has_value, offset, token_end);
      const bool can_advance = offset < text.size();
      const size_t safe_index =
        select_size(can_advance, offset, position);
      const char next_char = text[safe_index];
      const auto * next_walk = walk->traverse(next_char);
      walk = select_ptr(
        can_advance, next_walk, static_cast<decltype(next_walk)>(nullptr));
      offset += static_cast<size_t>(can_advance);
    }

    const bool emit_token = token_id != emel::text::encoders::detail::k_token_null;
    const bool token_push_ok = rwkv_push_token(ev.event_.request, token_id, count);
    const bool push_failed = emit_token && !token_push_ok;
    ev.event_.ctx.err = emel::text::encoders::rwkv::detail::select_i32(
      push_failed, EMEL_ERR_INVALID_ARGUMENT, ev.event_.ctx.err);
    active = active && !push_failed;
    position = token_end;
  }

  ev.event_.ctx.token_count = emel::text::encoders::rwkv::detail::select_i32(
    ev.event_.ctx.err == EMEL_OK, count, 0);
}

}  // namespace detail

struct begin_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.unk_lookup_found = false;
  }
};

struct begin_encode_sync_vocab {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    emel::text::encoders::action::sync_vocab(ev.event_, ctx);
    ctx.rwkv_tables_ready = false;
    ctx.rwkv_vocab = nullptr;
    ctx.token_matcher = emel::text::encoders::detail::naive_trie{};
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.unk_lookup_found = false;
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
    ev.unk_lookup_found = ev.unk_id != emel::text::encoders::detail::k_token_null;
  }
};

struct lookup_unk_candidate {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const detail::unk_lookup_result result = detail::lookup_unk_candidate(*ctx.vocab);
    ev.unk_id = result.id;
    ev.unk_lookup_found = result.found;
  }
};

struct set_unk_from_lookup {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.unk_id = emel::text::encoders::rwkv::detail::select_i32(
      ev.unk_lookup_found, ev.unk_id, emel::text::encoders::detail::k_token_null);
  }
};

struct set_unk_missing {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.unk_id = emel::text::encoders::detail::k_token_null;
    ev.unk_lookup_found = false;
  }
};

struct run_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    detail::run_encode_tokens(ev, ctx);
  }
};

struct sync_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::rwkv::detail::ensure_rwkv_tables(ctx, *ctx.vocab);
    ev.event_.ctx.err = emel::text::encoders::rwkv::detail::select_i32(
      ready, EMEL_OK, EMEL_ERR_INVALID_ARGUMENT);
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
inline constexpr lookup_unk_candidate lookup_unk_candidate{};
inline constexpr set_unk_from_lookup set_unk_from_lookup{};
inline constexpr set_unk_missing set_unk_missing{};
inline constexpr run_encode run_encode{};
inline constexpr sync_tables sync_tables{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::rwkv::action
