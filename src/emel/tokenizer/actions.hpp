#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "emel/emel.h"
#include "emel/tokenizer/context.hpp"

namespace emel::tokenizer::action {

template <class sm_type>
inline bool process_encoder(void *handle,
                            const emel::encoder::event::encode &ev) {
  if (handle == nullptr) {
    return false;
  }
  return static_cast<sm_type *>(handle)->process_event(ev);
}

template <class sm_type> inline encoder_entry make_encoder_entry(sm_type &sm_value) {
  return encoder_entry{&sm_value, process_encoder<sm_type>};
}
inline context::context()
    : bpe_encoder(bpe_ctx), spm_encoder(spm_ctx), wpm_encoder(wpm_ctx),
      ugm_encoder(ugm_ctx), rwkv_encoder(rwkv_ctx),
      plamo2_encoder(plamo2_ctx), fallback_encoder(fallback_ctx) {
  encoder_map[static_cast<size_t>(encoder_slot::spm)] =
      make_encoder_entry(spm_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::bpe)] =
      make_encoder_entry(bpe_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::wpm)] =
      make_encoder_entry(wpm_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::ugm)] =
      make_encoder_entry(ugm_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::rwkv)] =
      make_encoder_entry(rwkv_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::plamo2)] =
      make_encoder_entry(plamo2_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::fallback)] =
      make_encoder_entry(fallback_encoder);
  encoder_map[static_cast<size_t>(encoder_slot::none)] =
      encoder_map[static_cast<size_t>(encoder_slot::fallback)];
  active_encoder = &encoder_map[static_cast<size_t>(encoder_slot::none)];
}
} // namespace emel::tokenizer::action

namespace emel::tokenizer::detail {

using action::encoder_slot;
using action::fragment;
using action::fragment_kind;
using action::special_token;

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;

inline bool token_type_is_special(const int32_t type) {
  return type == k_token_type_control || type == k_token_type_user_defined ||
         type == k_token_type_unknown;
}

inline bool token_type_skip_when_no_parse(const int32_t type) {
  return type == k_token_type_control || type == k_token_type_unknown;
}

inline encoder_slot encoder_slot_from_model(
    const emel::model::data::tokenizer_model model) {
  switch (model) {
    case emel::model::data::tokenizer_model::SPM:
      return encoder_slot::spm;
    case emel::model::data::tokenizer_model::BPE:
      return encoder_slot::bpe;
    case emel::model::data::tokenizer_model::WPM:
      return encoder_slot::wpm;
    case emel::model::data::tokenizer_model::UGM:
      return encoder_slot::ugm;
    case emel::model::data::tokenizer_model::RWKV:
      return encoder_slot::rwkv;
    case emel::model::data::tokenizer_model::PLAMO2:
      return encoder_slot::plamo2;
    case emel::model::data::tokenizer_model::NONE:
    case emel::model::data::tokenizer_model::UNKNOWN:
    default:
      return encoder_slot::none;
  }
}

inline std::string_view token_text(const emel::model::data::vocab &vocab,
                                   uint32_t id) {
  if (id >= vocab.n_tokens) {
    return {};
  }
  const auto &entry = vocab.entries[id];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset,
                          entry.text_length);
}

inline bool
flag_set(const emel::model::data::vocab &vocab,
         const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes>
             &flags,
         const uint32_t id) {
  if (id >= vocab.n_tokens) {
    return false;
  }
  const uint32_t byte = id >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
  return (flags[byte] & mask) != 0;
}

inline bool has_lstrip(const emel::model::data::vocab &vocab,
                       const uint32_t id) {
  return flag_set(vocab, vocab.lstrip_flags, id);
}

inline bool has_rstrip(const emel::model::data::vocab &vocab,
                       const uint32_t id) {
  return flag_set(vocab, vocab.rstrip_flags, id);
}

inline bool is_special_type(const emel::model::data::vocab &vocab,
                            uint32_t id) {
  if (id >= vocab.n_tokens) {
    return false;
  }
  return token_type_is_special(vocab.entries[id].type);
}

inline bool append_token(action::context &ctx, const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (ctx.token_ids_out == nullptr) {
    return false;
  }
  if (ctx.token_capacity <= 0 || ctx.token_count >= ctx.token_capacity) {
    return false;
  }
  ctx.token_ids_out[ctx.token_count] = token;
  ctx.token_count += 1;
  return true;
}

inline bool push_raw_fragment(action::context &ctx,
                              const std::string_view text) {
  if (text.empty()) {
    return true;
  }
  if (ctx.fragment_count >= ctx.fragments.size()) {
    return false;
  }
  fragment &entry = ctx.fragments[ctx.fragment_count];
  entry.kind = action::fragment_kind::raw_text;
  entry.text = text;
  entry.token = -1;
  ctx.fragment_count += 1;
  return true;
}

inline bool push_token_fragment(action::context &ctx, const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (ctx.fragment_count >= ctx.fragments.size()) {
    return false;
  }
  fragment &entry = ctx.fragments[ctx.fragment_count];
  entry.kind = action::fragment_kind::token;
  entry.text = {};
  entry.token = token;
  ctx.fragment_count += 1;
  return true;
}

inline bool build_special_tokens(action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  if (ctx.special_vocab == &vocab && ctx.special_token_count > 0) {
    return true;
  }
  ctx.special_vocab = &vocab;
  ctx.special_token_count = 0;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    if (!is_special_type(vocab, i)) {
      continue;
    }
    const std::string_view text = token_text(vocab, i);
    if (text.empty()) {
      continue;
    }
    if (ctx.special_token_count >= ctx.special_tokens.size()) {
      return false;
    }
    special_token &entry = ctx.special_tokens[ctx.special_token_count];
    entry.text = text;
    entry.token = static_cast<int32_t>(i);
    entry.type = vocab.entries[i].type;
    entry.lstrip = has_lstrip(vocab, i);
    entry.rstrip = has_rstrip(vocab, i);
    ctx.special_token_count += 1;
  }
  std::sort(ctx.special_tokens.begin(),
            ctx.special_tokens.begin() +
                static_cast<std::ptrdiff_t>(ctx.special_token_count),
            [](const special_token &a, const special_token &b) {
              return a.text.size() > b.text.size();
            });
  return true;
}

struct special_match {
  bool found = false;
  size_t pos = 0;
  size_t len = 0;
  int32_t token = -1;
  bool lstrip = false;
  bool rstrip = false;
};

inline special_match find_next_special(const std::string_view text,
                                       const action::context &ctx,
                                       const bool parse_special) {
  special_match best = {};
  for (size_t i = 0; i < ctx.special_token_count; ++i) {
    const action::special_token &token = ctx.special_tokens[i];
    if (token.text.empty()) {
      continue;
    }
    if (!parse_special) {
      if (token_type_skip_when_no_parse(token.type)) {
        continue;
      }
    }
    const size_t pos = text.find(token.text);
    if (pos == std::string_view::npos) {
      continue;
    }
    if (!best.found || pos < best.pos ||
        (pos == best.pos && token.text.size() > best.len)) {
      best.found = true;
      best.pos = pos;
      best.len = token.text.size();
      best.token = token.token;
      best.lstrip = token.lstrip;
      best.rstrip = token.rstrip;
    }
  }
  return best;
}

} // namespace emel::tokenizer::detail

namespace emel::tokenizer::action {

inline void set_error(context &ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void reset_encoder_contexts(context &ctx,
                                   const emel::model::data::vocab *vocab) noexcept {
  auto reset_ctx = [vocab](auto &encoder_ctx) {
    if (encoder_ctx.vocab != vocab) {
      encoder_ctx.vocab = vocab;
      encoder_ctx.tables_ready = false;
      encoder_ctx.ugm_ready = false;
      using ctx_type = std::decay_t<decltype(encoder_ctx)>;
      if constexpr (std::is_same_v<ctx_type, emel::encoder::bpe::action::context>) {
        encoder_ctx.bpe_pre_id = emel::model::data::tokenizer_pre::DEFAULT;
        encoder_ctx.bpe_regex_exprs.clear();
      } else if constexpr (std::is_same_v<ctx_type, emel::encoder::ugm::action::context>) {
        encoder_ctx.ugm_tables_ready = false;
        encoder_ctx.ugm_vocab = nullptr;
        encoder_ctx.token_matcher = emel::encoder::detail::naive_trie{};
        encoder_ctx.user_defined_token_matcher = emel::encoder::detail::naive_trie{};
      } else if constexpr (std::is_same_v<ctx_type, emel::encoder::rwkv::action::context>) {
        encoder_ctx.rwkv_tables_ready = false;
        encoder_ctx.rwkv_vocab = nullptr;
        encoder_ctx.token_matcher = emel::encoder::detail::naive_trie{};
      } else if constexpr (std::is_same_v<ctx_type, emel::encoder::plamo2::action::context>) {
        encoder_ctx.plamo2_tables_ready = false;
        encoder_ctx.plamo2_vocab = nullptr;
        encoder_ctx.byte_tokens.fill(0);
        encoder_ctx.suffix_map.clear();
        encoder_ctx.table.clear();
      }
    }
  };
  reset_ctx(ctx.bpe_ctx);
  reset_ctx(ctx.spm_ctx);
  reset_ctx(ctx.wpm_ctx);
  reset_ctx(ctx.ugm_ctx);
  reset_ctx(ctx.rwkv_ctx);
  reset_ctx(ctx.plamo2_ctx);
  reset_ctx(ctx.fallback_ctx);
}

inline void clear_request(context &ctx) noexcept {
  ctx.vocab = nullptr;
  ctx.text = {};
  ctx.add_special = false;
  ctx.parse_special = false;
  ctx.token_ids_out = nullptr;
  ctx.token_capacity = 0;
  ctx.model_slot = encoder_slot::none;
}

struct begin_tokenize {
  void operator()(const event::tokenize &ev, context &ctx) const noexcept {
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    const auto * prior_vocab = ctx.vocab;
    ctx.vocab = ev.vocab;
    ctx.text = ev.text;
    ctx.add_special = ev.add_special;
    ctx.parse_special = ev.parse_special;
    ctx.token_ids_out = ev.token_ids_out;
    ctx.token_capacity = ev.token_capacity;
    ctx.model_slot = encoder_slot::none;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (prior_vocab != ev.vocab) {
      reset_encoder_contexts(ctx, ev.vocab);
    }
  }
};

struct reject_invalid {
  void operator()(const event::tokenize &, context &ctx) const noexcept {
    ctx.token_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct build_special_tokens {
  void operator()(context &ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.vocab == nullptr ||
        !detail::build_special_tokens(ctx, *ctx.vocab)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_raw {
  void operator()(context &ctx) const {
    ctx.phase_error = EMEL_OK;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (!detail::push_raw_fragment(ctx, ctx.text)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_with_specials {
  void operator()(context &ctx) const {
    ctx.phase_error = EMEL_OK;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (!detail::push_raw_fragment(ctx, ctx.text)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    std::array<fragment, k_max_fragments> next_fragments = {};
    for (size_t token_idx = 0; token_idx < ctx.special_token_count; ++token_idx) {
      const special_token &token = ctx.special_tokens[token_idx];
      if (token.text.empty()) {
        continue;
      }
      if (!ctx.parse_special && detail::token_type_skip_when_no_parse(token.type)) {
        continue;
      }

      size_t next_count = 0;
      auto push_raw = [&](const std::string_view text) {
        if (text.empty()) {
          return true;
        }
        if (next_count >= next_fragments.size()) {
          return false;
        }
        fragment &entry = next_fragments[next_count++];
        entry.kind = fragment_kind::raw_text;
        entry.text = text;
        entry.token = -1;
        return true;
      };
      auto push_token = [&](const int32_t token_id) {
        if (token_id < 0) {
          return false;
        }
        if (next_count >= next_fragments.size()) {
          return false;
        }
        fragment &entry = next_fragments[next_count++];
        entry.kind = fragment_kind::token;
        entry.text = {};
        entry.token = token_id;
        return true;
      };

      for (size_t frag_idx = 0; frag_idx < ctx.fragment_count; ++frag_idx) {
        const fragment &frag = ctx.fragments[frag_idx];
        if (frag.kind != fragment_kind::raw_text) {
          if (!push_token(frag.token)) {
            set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
            return;
          }
          continue;
        }

        const std::string_view raw = frag.text;
        size_t base_offset = 0;
        while (base_offset < raw.size()) {
          const size_t match = raw.find(token.text, base_offset);
          if (match == std::string_view::npos) {
            if (!push_raw(raw.substr(base_offset))) {
              set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
              return;
            }
            break;
          }
          size_t left_len = match - base_offset;
          if (token.lstrip) {
            while (left_len > 0 &&
                   std::isspace(static_cast<unsigned char>(
                       raw[base_offset + left_len - 1])) != 0) {
              left_len -= 1;
            }
          }
          if (left_len > 0) {
            if (!push_raw(raw.substr(base_offset, left_len))) {
              set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
              return;
            }
          }
          if (!push_token(token.token)) {
            set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
            return;
          }
          size_t right_offset = match + token.text.size();
          if (token.rstrip) {
            while (right_offset < raw.size() &&
                   std::isspace(static_cast<unsigned char>(
                       raw[right_offset])) != 0) {
              right_offset += 1;
            }
          }
          base_offset = right_offset;
        }
      }

      ctx.fragment_count = next_count;
      ctx.fragments = next_fragments;
      ctx.fragment_index = 0;
    }
  }
};

struct select_backend {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    reset_encoder_contexts(ctx, ctx.vocab);
    const auto slot = detail::encoder_slot_from_model(
        ctx.vocab->tokenizer_model_id);
    ctx.model_slot = slot;
    ctx.active_encoder = &ctx.encoder_map[static_cast<size_t>(slot)];
    if (ctx.active_encoder == nullptr ||
        ctx.active_encoder->process == nullptr) {
      set_error(ctx, EMEL_ERR_MODEL_INVALID);
    }
  }
};

struct append_bos {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (!detail::append_token(ctx, ctx.vocab->bos_id)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct append_sep {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (!detail::append_token(ctx, ctx.vocab->sep_id)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct append_eos {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (!detail::append_token(ctx, ctx.vocab->eos_id)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct append_fragment_token {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.fragment_index >= ctx.fragment_count) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const fragment &frag = ctx.fragments[ctx.fragment_index];
    if (frag.kind != fragment_kind::token) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (!detail::append_token(ctx, frag.token)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    ctx.fragment_index += 1;
  }
};

struct encode_raw_fragment {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.fragment_index >= ctx.fragment_count) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const fragment &frag = ctx.fragments[ctx.fragment_index];
    if (frag.kind != fragment_kind::raw_text) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.active_encoder == nullptr ||
        ctx.active_encoder->process == nullptr) {
      set_error(ctx, EMEL_ERR_MODEL_INVALID);
      return;
    }
    if (ctx.token_ids_out == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const int32_t capacity = ctx.token_capacity - ctx.token_count;
    if (capacity < 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t fragment_count = 0;
    int32_t err = EMEL_OK;
    emel::encoder::event::encode encode_ev = {};
    encode_ev.text = frag.text;
    encode_ev.token_ids = ctx.token_ids_out + ctx.token_count;
    encode_ev.token_capacity = capacity;
    encode_ev.token_count_out = &fragment_count;
    encode_ev.error_out = &err;
    ctx.active_encoder->process(ctx.active_encoder->handle, encode_ev);
    if (err != EMEL_OK) {
      set_error(ctx, err);
      return;
    }
    if (fragment_count < 0 || fragment_count > capacity) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    ctx.token_count += fragment_count;
    ctx.fragment_index += 1;
  }
};

struct finalize {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct set_capacity_error {
  void operator()(context &ctx) const noexcept {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct set_invalid_id_error {
  void operator()(context &ctx) const noexcept {
    set_error(ctx, EMEL_ERR_MODEL_INVALID);
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event &, context &ctx) const noexcept {
    ctx.token_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_tokenize begin_tokenize{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr build_special_tokens build_special_tokens{};
inline constexpr partition_raw partition_raw{};
inline constexpr partition_with_specials partition_with_specials{};
inline constexpr select_backend select_backend{};
inline constexpr append_bos append_bos{};
inline constexpr append_sep append_sep{};
inline constexpr append_eos append_eos{};
inline constexpr append_fragment_token append_fragment_token{};
inline constexpr encode_raw_fragment encode_raw_fragment{};
inline constexpr finalize finalize{};
inline constexpr set_capacity_error set_capacity_error{};
inline constexpr set_invalid_id_error set_invalid_id_error{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::tokenizer::action
