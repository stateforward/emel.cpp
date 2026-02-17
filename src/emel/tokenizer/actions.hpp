#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cctype>
#include <cstdint>
#include <string_view>

#include "boost/sml.hpp"
#include "emel/emel.h"
#include "emel/encoder/actions.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/tokenizer/events.hpp"

namespace emel::tokenizer {

using process_t = boost::sml::back::process<
  event::special_tokens_ready,
  event::partitioning_special_done,
  event::partitioning_special_error,
  event::selecting_backend_done,
  event::selecting_backend_error,
  event::applying_special_prefix_done,
  event::applying_special_prefix_error,
  event::encoding_fragment_done,
  event::encoding_fragment_error,
  event::next_fragment,
  event::applying_special_suffix_done,
  event::applying_special_suffix_error,
  event::finalizing_done,
  event::finalizing_error>;

}  // namespace emel::tokenizer

namespace emel::tokenizer::action {

constexpr size_t k_max_fragments = 1024;
constexpr size_t k_max_special_tokens = 256;

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
  emel::encoder::sm encoder;
  std::array<fragment, k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  std::array<special_token, k_max_special_tokens> special_tokens = {};
  size_t special_token_count = 0;
  const emel::model::data::vocab * special_vocab = nullptr;
  int32_t token_count = 0;

  context() : encoder(encoder_ctx) {}
};

}  // namespace emel::tokenizer::action

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

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;

inline bool token_type_is_special(const int32_t type) {
  return type == k_token_type_control ||
         type == k_token_type_user_defined ||
         type == k_token_type_unknown;
}

inline bool token_type_skip_when_no_parse(const int32_t type) {
  return type == k_token_type_control || type == k_token_type_unknown;
}

template <size_t N>
inline std::string_view string_view_from_array(const std::array<char, N> & data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline tokenizer_model detect_model(const emel::model::data::vocab & vocab) {
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

inline std::string_view token_text(const emel::model::data::vocab & vocab, uint32_t id) {
  if (id >= vocab.n_tokens) {
    return {};
  }
  const auto & entry = vocab.entries[id];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset, entry.text_length);
}

inline bool flag_set(
    const emel::model::data::vocab & vocab,
    const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const uint32_t id) {
  if (id >= vocab.n_tokens) {
    return false;
  }
  const uint32_t byte = id >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
  return (flags[byte] & mask) != 0;
}

inline bool has_lstrip(const emel::model::data::vocab & vocab, const uint32_t id) {
  return flag_set(vocab, vocab.lstrip_flags, id);
}

inline bool has_rstrip(const emel::model::data::vocab & vocab, const uint32_t id) {
  return flag_set(vocab, vocab.rstrip_flags, id);
}

inline bool is_special_type(const emel::model::data::vocab & vocab, uint32_t id) {
  if (id >= vocab.n_tokens) {
    return false;
  }
  return token_type_is_special(vocab.entries[id].type);
}

inline bool append_token(
    const event::tokenize & ev,
    action::context & ctx,
    const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (ev.token_ids_out == nullptr || ev.token_count_out == nullptr) {
    return false;
  }
  if (ev.token_capacity <= 0 || ctx.token_count >= ev.token_capacity) {
    return false;
  }
  ev.token_ids_out[ctx.token_count] = token;
  ctx.token_count += 1;
  *ev.token_count_out = ctx.token_count;
  return true;
}

inline bool push_raw_fragment(action::context & ctx, const std::string_view text) {
  if (text.empty()) {
    return true;
  }
  if (ctx.fragment_count >= ctx.fragments.size()) {
    return false;
  }
  fragment & entry = ctx.fragments[ctx.fragment_count];
  entry.kind = action::fragment_kind::raw_text;
  entry.text = text;
  entry.token = -1;
  ctx.fragment_count += 1;
  return true;
}

inline bool push_token_fragment(action::context & ctx, const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (ctx.fragment_count >= ctx.fragments.size()) {
    return false;
  }
  fragment & entry = ctx.fragments[ctx.fragment_count];
  entry.kind = action::fragment_kind::token;
  entry.text = {};
  entry.token = token;
  ctx.fragment_count += 1;
  return true;
}

inline bool build_special_tokens(action::context & ctx, const emel::model::data::vocab & vocab) {
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
    special_token & entry = ctx.special_tokens[ctx.special_token_count];
    entry.text = text;
    entry.token = static_cast<int32_t>(i);
    entry.type = vocab.entries[i].type;
    entry.lstrip = has_lstrip(vocab, i);
    entry.rstrip = has_rstrip(vocab, i);
    ctx.special_token_count += 1;
  }
  std::sort(
    ctx.special_tokens.begin(),
    ctx.special_tokens.begin() + static_cast<std::ptrdiff_t>(ctx.special_token_count),
    [](const special_token & a, const special_token & b) {
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

inline special_match find_next_special(
    const std::string_view text,
    const action::context & ctx,
    const bool parse_special) {
  special_match best = {};
  for (size_t i = 0; i < ctx.special_token_count; ++i) {
    const action::special_token & token = ctx.special_tokens[i];
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
    if (!best.found || pos < best.pos || (pos == best.pos && token.text.size() > best.len)) {
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

template <class Event>
inline const event::tokenize * request_from(const Event &) {
  return nullptr;
}

inline const event::tokenize * request_from(const event::tokenize & ev) { return &ev; }
inline const event::tokenize * request_from(const event::special_tokens_ready & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::partitioning_special_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::partitioning_special_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::selecting_backend_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::selecting_backend_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::applying_special_prefix_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::applying_special_prefix_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::encoding_fragment_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::encoding_fragment_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::next_fragment & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::applying_special_suffix_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::applying_special_suffix_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::finalizing_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const event::finalizing_error & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const events::tokenizer_done & ev) {
  return ev.request;
}
inline const event::tokenize * request_from(const events::tokenizer_error & ev) {
  return ev.request;
}

}  // namespace emel::tokenizer::detail

namespace emel::tokenizer::action {

struct begin_tokenize {
  void operator()(const event::tokenize & ev, action::context & ctx) const {
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    ctx.token_count = 0;
    if (ctx.encoder_ctx.vocab != ev.vocab) {
      ctx.encoder_ctx.vocab = ev.vocab;
      ctx.encoder_ctx.tables_ready = false;
      ctx.encoder_ctx.ugm_ready = false;
    }
  }
};

struct reject_invalid {
  void operator()(const event::tokenize & ev, action::context &) const {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
  }
};

struct partition_special {
  void operator()(const event::tokenize & ev, action::context & ctx, process_t & process) const {
    ctx.fragment_count = 0;
    ctx.fragment_index = 0;
    if (!detail::build_special_tokens(ctx, *ev.vocab)) {
      process(event::partitioning_special_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    process(event::special_tokens_ready{&ev});
  }
};

struct partition_raw {
  void operator()(const event::special_tokens_ready & ev, action::context & ctx, process_t & process) const {
    const event::tokenize * request = ev.request;
    if (!detail::push_raw_fragment(ctx, request->text)) {
      process(event::partitioning_special_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    process(event::partitioning_special_done{request});
  }
};

struct partition_with_specials {
  void operator()(const event::special_tokens_ready & ev, action::context & ctx, process_t & process) const {
    const event::tokenize * request = ev.request;
    size_t offset = 0;
    while (offset < request->text.size()) {
      const std::string_view remaining = request->text.substr(offset);
      const detail::special_match match =
        detail::find_next_special(remaining, ctx, request->parse_special);
      if (!match.found) {
        if (!detail::push_raw_fragment(ctx, remaining)) {
          process(event::partitioning_special_error{request, EMEL_ERR_INVALID_ARGUMENT});
          return;
        }
        break;
      }
      size_t left_len = match.pos;
      if (match.lstrip) {
        while (left_len > 0 &&
               std::isspace(static_cast<unsigned char>(remaining[left_len - 1])) != 0) {
          left_len -= 1;
        }
      }
      if (left_len > 0) {
        const std::string_view left = remaining.substr(0, left_len);
        if (!detail::push_raw_fragment(ctx, left)) {
          process(event::partitioning_special_error{request, EMEL_ERR_INVALID_ARGUMENT});
          return;
        }
      }
      if (!detail::push_token_fragment(ctx, match.token)) {
        process(event::partitioning_special_error{request, EMEL_ERR_INVALID_ARGUMENT});
        return;
      }
      size_t right_offset = match.pos + match.len;
      if (match.rstrip) {
        while (right_offset < remaining.size() &&
               std::isspace(static_cast<unsigned char>(remaining[right_offset])) != 0) {
          right_offset += 1;
        }
      }
      offset += right_offset;
    }
    process(event::partitioning_special_done{request});
  }
};

struct select_backend {
  void operator()(const event::partitioning_special_done & ev, action::context & ctx, process_t & process) const {
    const event::tokenize * request = ev.request;
    if (ctx.encoder_ctx.vocab != request->vocab) {
      ctx.encoder_ctx.vocab = request->vocab;
      ctx.encoder_ctx.tables_ready = false;
      ctx.encoder_ctx.ugm_ready = false;
    }
    process(event::selecting_backend_done{request});
  }
};

struct append_bos {
  void operator()(const event::selecting_backend_done & ev, action::context & ctx) const {
    const event::tokenize * request = ev.request;
    (void)detail::append_token(*request, ctx, request->vocab->bos_id);
  }
};

struct emit_prefix_invalid_id_error {
  void operator()(const event::selecting_backend_done & ev, process_t & process) const {
    process(event::applying_special_prefix_error{ev.request, EMEL_ERR_MODEL_INVALID});
  }
};

struct emit_prefix_capacity_error {
  void operator()(const event::selecting_backend_done & ev, process_t & process) const {
    process(event::applying_special_prefix_error{ev.request, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct dispatch_next_fragment {
  template <class Event>
  void operator()(const Event & ev, process_t & process) const {
    const event::tokenize * request = detail::request_from(ev);
    process(event::next_fragment{request});
  }
};

struct encode_fragment {
  void operator()(const event::next_fragment & ev, action::context & ctx, process_t & process) const {
    const event::tokenize * request = ev.request;
    const fragment & frag = ctx.fragments[ctx.fragment_index];
    if (frag.kind == fragment_kind::token) {
      (void)detail::append_token(*request, ctx, frag.token);
      ctx.fragment_index += 1;
      process(event::encoding_fragment_done{request});
      return;
    }
    int32_t fragment_count = 0;
    int32_t err = EMEL_OK;
    const int32_t capacity = request->token_capacity - ctx.token_count;
    emel::encoder::event::encode encode_ev = {};
    encode_ev.text = frag.text;
    encode_ev.token_ids = request->token_ids_out + ctx.token_count;
    encode_ev.token_capacity = capacity;
    encode_ev.token_count_out = &fragment_count;
    encode_ev.error_out = &err;
    ctx.encoder.process_event(encode_ev);
    if (err != EMEL_OK) {
      process(event::encoding_fragment_error{request, err});
      return;
    }
    ctx.token_count += fragment_count;
    if (request->token_count_out != nullptr) {
      *request->token_count_out = ctx.token_count;
    }
    ctx.fragment_index += 1;
    process(event::encoding_fragment_done{request});
  }
};

struct dispatch_no_fragment_done {
  void operator()(const event::next_fragment & ev, process_t & process) const {
    process(event::encoding_fragment_done{ev.request});
  }
};

struct dispatch_capacity_error {
  void operator()(const event::next_fragment & ev, process_t & process) const {
    process(event::encoding_fragment_error{ev.request, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct append_sep {
  void operator()(const event::encoding_fragment_done & ev, action::context & ctx) const {
    const event::tokenize * request = ev.request;
    (void)detail::append_token(*request, ctx, request->vocab->sep_id);
  }
};

struct append_eos {
  void operator()(const event::encoding_fragment_done & ev, action::context & ctx) const {
    const event::tokenize * request = ev.request;
    (void)detail::append_token(*request, ctx, request->vocab->eos_id);
  }
};

struct emit_suffix_invalid_id_error {
  void operator()(const event::encoding_fragment_done & ev, process_t & process) const {
    process(event::applying_special_suffix_error{ev.request, EMEL_ERR_MODEL_INVALID});
  }
};

struct emit_suffix_capacity_error {
  void operator()(const event::encoding_fragment_done & ev, process_t & process) const {
    process(event::applying_special_suffix_error{ev.request, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct finalize {
  template <class Event>
  void operator()(const Event & ev, action::context & ctx, process_t & process) const {
    const event::tokenize * request = detail::request_from(ev);
    *request->token_count_out = ctx.token_count;
    process(event::finalizing_done{request});
  }
};

struct dispatch_done {
  void operator()(const event::finalizing_done & ev, action::context & ctx, process_t &) const {
    const event::tokenize * request = ev.request;
    if (request == nullptr || request->dispatch_done == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_done(request->owner_sm, events::tokenizer_done{request, ctx.token_count});
  }
};

struct dispatch_reject {
  void operator()(const event::tokenize & ev, process_t &) const {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.dispatch_error == nullptr || ev.owner_sm == nullptr) {
      return;
    }
    ev.dispatch_error(ev.owner_sm, events::tokenizer_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct dispatch_error {
  template <class ErrorEvent>
  void operator()(const ErrorEvent & ev, process_t &) const {
    const event::tokenize * request = ev.request;
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = ev.err;
    }
    if (request->dispatch_error == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_error(request->owner_sm, events::tokenizer_error{request, ev.err});
  }
};

struct dispatch_unexpected {
  template <class Event>
  void operator()(const Event & ev, process_t &) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(
        request->owner_sm,
        events::tokenizer_error{request, EMEL_ERR_INVALID_ARGUMENT});
    }
  }
};

inline constexpr begin_tokenize begin_tokenize{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr partition_special partition_special{};
inline constexpr partition_raw partition_raw{};
inline constexpr partition_with_specials partition_with_specials{};
inline constexpr select_backend select_backend{};
inline constexpr append_bos append_bos{};
inline constexpr emit_prefix_invalid_id_error emit_prefix_invalid_id_error{};
inline constexpr emit_prefix_capacity_error emit_prefix_capacity_error{};
inline constexpr dispatch_next_fragment dispatch_next_fragment{};
inline constexpr encode_fragment encode_fragment{};
inline constexpr dispatch_no_fragment_done dispatch_no_fragment_done{};
inline constexpr dispatch_capacity_error dispatch_capacity_error{};
inline constexpr append_sep append_sep{};
inline constexpr append_eos append_eos{};
inline constexpr emit_suffix_invalid_id_error emit_suffix_invalid_id_error{};
inline constexpr emit_suffix_capacity_error emit_suffix_capacity_error{};
inline constexpr finalize finalize{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_reject dispatch_reject{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr dispatch_unexpected dispatch_unexpected{};

}  // namespace emel::tokenizer::action
