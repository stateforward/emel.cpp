#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/encoder/events.hpp"
#include "emel/tokenizer/context.hpp"
#include "emel/tokenizer/preprocessor/events.hpp"

namespace emel::tokenizer::action {

inline context::context()
    : encoder_any() {
  preprocessor_any.set_kind(preprocessor_kind::fallback);
  encoder_any.set_kind(encoder_kind::fallback);
  preprocess_kind = preprocessor_kind::fallback;
  model_kind = encoder_kind::fallback;
  is_bound = false;
}
} // namespace emel::tokenizer::action

namespace emel::tokenizer::detail {

using action::encoder_kind;
using action::preprocessor_kind;

inline encoder_kind encoder_kind_from_model(
    const emel::model::data::tokenizer_model model) {
  switch (model) {
    case emel::model::data::tokenizer_model::SPM:
      return encoder_kind::spm;
    case emel::model::data::tokenizer_model::BPE:
      return encoder_kind::bpe;
    case emel::model::data::tokenizer_model::WPM:
      return encoder_kind::wpm;
    case emel::model::data::tokenizer_model::UGM:
      return encoder_kind::ugm;
    case emel::model::data::tokenizer_model::RWKV:
      return encoder_kind::rwkv;
    case emel::model::data::tokenizer_model::PLAMO2:
      return encoder_kind::plamo2;
    case emel::model::data::tokenizer_model::NONE:
    case emel::model::data::tokenizer_model::UNKNOWN:
    default:
      return encoder_kind::fallback;
  }
}

inline preprocessor_kind preprocessor_kind_from_model(
    const emel::model::data::tokenizer_model model) {
  switch (model) {
    case emel::model::data::tokenizer_model::SPM:
      return preprocessor_kind::spm;
    case emel::model::data::tokenizer_model::BPE:
      return preprocessor_kind::bpe;
    case emel::model::data::tokenizer_model::WPM:
      return preprocessor_kind::wpm;
    case emel::model::data::tokenizer_model::UGM:
      return preprocessor_kind::ugm;
    case emel::model::data::tokenizer_model::RWKV:
      return preprocessor_kind::rwkv;
    case emel::model::data::tokenizer_model::PLAMO2:
      return preprocessor_kind::plamo2;
    case emel::model::data::tokenizer_model::NONE:
    case emel::model::data::tokenizer_model::UNKNOWN:
    default:
      return preprocessor_kind::fallback;
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

inline void clear_request(context &ctx) noexcept {
  ctx.text = {};
  ctx.add_special = false;
  ctx.parse_special = false;
  ctx.token_ids_out = nullptr;
  ctx.token_capacity = 0;
}

struct begin_bind {
  void operator()(const event::bind &ev, context &ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.vocab = ev.vocab;
    ctx.is_bound = false;
    ctx.preprocess_kind = preprocessor_kind::fallback;
    ctx.model_kind = encoder_kind::fallback;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_bind {
  void operator()(const event::bind &, context &ctx) const noexcept {
    ctx.is_bound = false;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct begin_tokenize {
  void operator()(const event::tokenize &ev, context &ctx) const noexcept {
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.text = ev.text;
    ctx.add_special = ev.add_special;
    ctx.parse_special = ev.parse_special;
    ctx.token_ids_out = ev.token_ids_out;
    ctx.token_capacity = ev.token_capacity;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_invalid {
  void operator()(const event::tokenize &, context &ctx) const noexcept {
    ctx.token_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct bind_preprocessor {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const auto kind = detail::preprocessor_kind_from_model(
        ctx.vocab->tokenizer_model_id);
    ctx.preprocess_kind = kind;
    ctx.preprocessor_any.set_kind(kind);
  }
};

struct bind_encoder {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.is_bound = false;
    if (ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const auto kind = detail::encoder_kind_from_model(
        ctx.vocab->tokenizer_model_id);
    ctx.model_kind = kind;
    ctx.encoder_any.set_kind(kind);
    ctx.is_bound = true;
  }
};

struct run_preprocess {
  void operator()(context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    size_t fragment_count = 0;
    int32_t err = EMEL_OK;
    emel::tokenizer::preprocessor::event::preprocess ev = {};
    ev.vocab = ctx.vocab;
    ev.text = ctx.text;
    ev.parse_special = ctx.parse_special;
    ev.fragments_out = ctx.fragments.data();
    ev.fragment_capacity = ctx.fragments.size();
    ev.fragment_count_out = &fragment_count;
    ev.error_out = &err;
    const bool accepted = ctx.preprocessor_any.process_event(ev);
    if (!accepted && err == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    if (err != EMEL_OK) {
      set_error(ctx, err);
      return;
    }
    ctx.fragment_count = fragment_count;
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
    encode_ev.vocab = ctx.vocab;
    encode_ev.text = frag.text;
    encode_ev.pretokenized = (ctx.model_kind == encoder_kind::bpe);
    encode_ev.token_ids = ctx.token_ids_out + ctx.token_count;
    encode_ev.token_capacity = capacity;
    encode_ev.token_count_out = &fragment_count;
    encode_ev.error_out = &err;
    const bool accepted = ctx.encoder_any.process_event(encode_ev);
    if (!accepted && err == EMEL_OK) {
      set_error(ctx, EMEL_ERR_MODEL_INVALID);
      return;
    }
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

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr begin_tokenize begin_tokenize{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr bind_preprocessor bind_preprocessor{};
inline constexpr bind_encoder bind_encoder{};
inline constexpr run_preprocess run_preprocess{};
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
