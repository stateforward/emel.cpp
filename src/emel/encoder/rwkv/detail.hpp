#pragma once

#include <string>

#include "emel/encoder/rwkv/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::rwkv::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline bool unescape_rwkv_token(const std::string_view escaped,
                                std::string &out) {
  out.clear();
  out.reserve(escaped.size());
  bool escaping = false;
  uint8_t hex_remaining = 0;
  uint8_t hex_acc = 0;

  for (const char c : escaped) {
    if (hex_remaining != 0) {
      const uint8_t value = (c >= 'a') ? static_cast<uint8_t>(c - 'a' + 10)
                                       : static_cast<uint8_t>(c - '0');
      hex_acc = static_cast<uint8_t>((hex_acc << 4) + value);
      hex_remaining -= 1;
      if (hex_remaining == 0) {
        out.push_back(static_cast<char>(hex_acc));
        hex_acc = 0;
      }
      continue;
    }
    if (escaping) {
      if (c == 't') {
        out.push_back('\t');
      } else if (c == 'n') {
        out.push_back('\n');
      } else if (c == 'r') {
        out.push_back('\r');
      } else if (c == 'x') {
        hex_remaining = 2;
      } else {
        out.push_back(c);
      }
      escaping = false;
      continue;
    }
    if (c == '\\') {
      escaping = true;
      continue;
    }
    out.push_back(c);
  }
  return hex_remaining == 0;
}

inline bool ensure_rwkv_tables(emel::encoder::rwkv::action::context &ctx,
                               const emel::model::data::vocab &vocab) {
  if (ctx.rwkv_tables_ready && ctx.rwkv_vocab == &vocab) {
    return true;
  }
  ctx.rwkv_vocab = &vocab;
  ctx.rwkv_tables_ready = false;
  ctx.token_matcher = emel::encoder::detail::naive_trie{};

  std::string unescaped;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text = emel::encoder::detail::token_text(vocab, id);
    if (text.empty()) {
      continue;
    }
    if (!unescape_rwkv_token(text, unescaped)) {
      return false;
    }
    if (!unescaped.empty()) {
      ctx.token_matcher.insert(unescaped.data(), unescaped.size(),
                               static_cast<int32_t>(id));
    }
  }
  ctx.rwkv_tables_ready = true;
  return true;
}

inline encode_result encode_rwkv(const event::encode &ev,
                                 emel::encoder::rwkv::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  if (!ensure_rwkv_tables(ctx, vocab)) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  int32_t count = 0;
  const int32_t unk_id = (vocab.unk_id != k_token_null)
                             ? vocab.unk_id
                             : emel::encoder::detail::lookup_token(ctx, "<unk>");
  size_t position = 0;
  while (position < ev.text.size()) {
    const auto *node = ctx.token_matcher.traverse(ev.text[position]);
    if (node == nullptr) {
      if (unk_id != k_token_null &&
          !emel::encoder::detail::push_token(ev, unk_id, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      position += 1;
      continue;
    }
    int32_t token_id = unk_id;
    size_t token_length = position + 1;
    size_t offset = position + 1;
    while (node != nullptr) {
      if (node->has_value) {
        token_id = node->value;
        token_length = offset;
      }
      if (offset >= ev.text.size()) {
        break;
      }
      node = node->traverse(ev.text[offset]);
      offset += 1;
    }
    if (token_id != k_token_null) {
      if (!emel::encoder::detail::push_token(ev, token_id, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
    }
    position = token_length;
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::rwkv::detail
