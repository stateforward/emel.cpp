#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/encoder/actions.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/tokenizer/events.hpp"

namespace emel::tokenizer::action {

constexpr size_t k_max_fragments = 1024;
constexpr size_t k_max_special_tokens = 256;
constexpr size_t k_encoder_map_size = 8;

enum class encoder_slot : uint8_t {
  none = 0,
  spm = 1,
  bpe = 2,
  wpm = 3,
  ugm = 4,
  rwkv = 5,
  plamo2 = 6,
  fallback = 7,
};

struct encoder_entry {
  void *handle = nullptr;
  bool (*process)(void *handle,
                  const emel::encoder::event::encode &ev) = nullptr;
};

template <class Sm>
inline bool process_encoder(void *handle,
                            const emel::encoder::event::encode &ev) {
  if (handle == nullptr) {
    return false;
  }
  return static_cast<Sm *>(handle)->process_event(ev);
}

template <class Sm> inline encoder_entry make_encoder_entry(Sm &sm) {
  return encoder_entry{&sm, process_encoder<Sm>};
}

enum class fragment_kind : uint8_t {
  raw_text = 0,
  token = 1,
};

struct fragment {
  fragment_kind kind = fragment_kind::raw_text;
  std::string_view text = {};
  int32_t token = -1;
};

struct special_token {
  std::string_view text = {};
  int32_t token = -1;
  int32_t type = 0;
  bool lstrip = false;
  bool rstrip = false;
};

struct context {
  emel::encoder::action::context encoder_ctx = {};
  emel::encoder::bpe::sm bpe_encoder;
  emel::encoder::spm::sm spm_encoder;
  emel::encoder::wpm::sm wpm_encoder;
  emel::encoder::ugm::sm ugm_encoder;
  emel::encoder::rwkv::sm rwkv_encoder;
  emel::encoder::plamo2::sm plamo2_encoder;
  emel::encoder::fallback::sm fallback_encoder;
  std::array<encoder_entry, k_encoder_map_size> encoder_map = {};
  encoder_entry *active_encoder = nullptr;
  std::array<fragment, k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  std::array<special_token, k_max_special_tokens> special_tokens = {};
  size_t special_token_count = 0;
  const emel::model::data::vocab *special_vocab = nullptr;
  const emel::model::data::vocab *vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  int32_t *token_ids_out = nullptr;
  int32_t token_capacity = 0;
  encoder_slot model_slot = encoder_slot::none;
  int32_t token_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  context()
      : bpe_encoder(encoder_ctx), spm_encoder(encoder_ctx),
        wpm_encoder(encoder_ctx), ugm_encoder(encoder_ctx),
        rwkv_encoder(encoder_ctx), plamo2_encoder(encoder_ctx),
        fallback_encoder(encoder_ctx) {
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
};

} // namespace emel::tokenizer::action

namespace emel::tokenizer::detail {

enum class tokenizer_model {
  none = 0,
  spm = 1,
  bpe = 2,
  wpm = 3,
  ugm = 4,
  rwkv = 5,
  plamo2 = 6,
  unknown = 7,
};

static_assert(static_cast<size_t>(tokenizer_model::unknown) + 1 ==
                  action::k_encoder_map_size,
              "encoder map size must cover tokenizer models");

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

template <size_t N>
inline std::string_view
string_view_from_array(const std::array<char, N> &data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline tokenizer_model detect_model(const emel::model::data::vocab &vocab) {
  const std::string_view model = string_view_from_array(vocab.tokenizer_model);
  if (model.empty() || model == "none" || model == "no_vocab") {
    return tokenizer_model::none;
  }
  if (model == "llama") {
    return tokenizer_model::spm;
  }
  if (model == "bert") {
    return tokenizer_model::wpm;
  }
  if (model == "gpt2") {
    return tokenizer_model::bpe;
  }
  if (model == "t5") {
    return tokenizer_model::ugm;
  }
  if (model == "rwkv") {
    return tokenizer_model::rwkv;
  }
  if (model == "plamo2") {
    return tokenizer_model::plamo2;
  }
  return tokenizer_model::unknown;
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
    if (ctx.encoder_ctx.vocab != ev.vocab) {
      ctx.encoder_ctx.vocab = ev.vocab;
      ctx.encoder_ctx.tables_ready = false;
      ctx.encoder_ctx.ugm_ready = false;
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
    bool ok = true;
    size_t offset = 0;
    while (offset < ctx.text.size()) {
      const std::string_view remaining = ctx.text.substr(offset);
      const detail::special_match match =
          detail::find_next_special(remaining, ctx, ctx.parse_special);
      if (!match.found) {
        if (!detail::push_raw_fragment(ctx, remaining)) {
          ok = false;
        }
        break;
      }
      size_t left_len = match.pos;
      if (match.lstrip) {
        while (left_len > 0 && std::isspace(static_cast<unsigned char>(
                                   remaining[left_len - 1])) != 0) {
          left_len -= 1;
        }
      }
      if (left_len > 0) {
        const std::string_view left = remaining.substr(0, left_len);
        if (!detail::push_raw_fragment(ctx, left)) {
          ok = false;
          break;
        }
      }
      if (!detail::push_token_fragment(ctx, match.token)) {
        ok = false;
        break;
      }
      size_t right_offset = match.pos + match.len;
      if (match.rstrip) {
        while (right_offset < remaining.size() &&
               std::isspace(
                   static_cast<unsigned char>(remaining[right_offset])) != 0) {
          right_offset += 1;
        }
      }
      offset += right_offset;
    }
    if (!ok) {
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
    if (ctx.encoder_ctx.vocab != ctx.vocab) {
      ctx.encoder_ctx.vocab = ctx.vocab;
      ctx.encoder_ctx.tables_ready = false;
      ctx.encoder_ctx.ugm_ready = false;
    }
    const auto slot =
        static_cast<encoder_slot>(detail::detect_model(*ctx.vocab));
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
  template <class Event>
  void operator()(const Event &, context &ctx) const noexcept {
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
