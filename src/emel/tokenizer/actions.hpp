#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "emel/emel.h"
#include "emel/tokenizer/context.hpp"
#include "emel/tokenizer/preprocessor/detail.hpp"

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
        !emel::tokenizer::preprocessor::detail::build_special_tokens(
            ctx.special_cache, *ctx.vocab)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_raw {
  void operator()(context &ctx) const {
    ctx.phase_error = EMEL_OK;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (!emel::tokenizer::preprocessor::detail::push_raw_fragment(
            ctx.fragments.data(), ctx.fragments.size(), ctx.fragment_count,
            ctx.text)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_with_specials {
  void operator()(context &ctx) const {
    ctx.phase_error = EMEL_OK;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (!emel::tokenizer::preprocessor::detail::partition_with_specials(
            ctx.text, ctx.special_cache, ctx.parse_special,
            ctx.fragments.data(), ctx.fragments.size(), &ctx.fragment_count)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
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
