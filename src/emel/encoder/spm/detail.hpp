#pragma once

#include <cstring>
#include <limits>

#include "emel/encoder/spm/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::spm::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline encode_result encode_spm(const event::encode &ev,
                                emel::encoder::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  size_t out_len = 0;
  if (vocab.add_space_prefix && ev.text.front() != ' ') {
    if (out_len + 3 > ctx.scratch.buffer.size()) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
    ctx.scratch.buffer[out_len++] = '\xE2';
    ctx.scratch.buffer[out_len++] = '\x96';
    ctx.scratch.buffer[out_len++] = '\x81';
  }
  for (const char c : ev.text) {
    if (c == ' ') {
      if (out_len + 3 > ctx.scratch.buffer.size()) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      ctx.scratch.buffer[out_len++] = '\xE2';
      ctx.scratch.buffer[out_len++] = '\x96';
      ctx.scratch.buffer[out_len++] = '\x81';
    } else {
      if (out_len + 1 > ctx.scratch.buffer.size()) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      ctx.scratch.buffer[out_len++] = c;
    }
  }

  const std::string_view escaped(ctx.scratch.buffer.data(), out_len);
  if (!emel::encoder::detail::build_symbols(escaped, ctx.scratch, result)) {
    return result;
  }

  for (;;) {
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_left = -1;
    int32_t best_right = -1;
    for (int32_t left = 0; left != -1; left = ctx.scratch.next[static_cast<size_t>(left)]) {
      const int32_t right = ctx.scratch.next[static_cast<size_t>(left)];
      if (right < 0) {
        break;
      }
      const std::string_view left_view =
          escaped.substr(ctx.scratch.offsets[static_cast<size_t>(left)],
                         ctx.scratch.lengths[static_cast<size_t>(left)]);
      const std::string_view right_view =
          escaped.substr(ctx.scratch.offsets[static_cast<size_t>(right)],
                         ctx.scratch.lengths[static_cast<size_t>(right)]);
      const int32_t token = emel::encoder::detail::lookup_token_concat(ctx, left_view, right_view);
      if (token == k_token_null) {
        continue;
      }
      const float score = vocab.entries[static_cast<uint32_t>(token)].score;
      if (score > best_score || (score == best_score && left < best_left)) {
        best_score = score;
        best_left = left;
        best_right = right;
      }
    }
    if (best_left < 0 || best_right < 0) {
      break;
    }
    emel::encoder::detail::merge_symbols(ctx.scratch, best_left, best_right);
  }

  int32_t count = 0;
  for (int32_t idx = 0; idx != -1; idx = ctx.scratch.next[static_cast<size_t>(idx)]) {
    if (ctx.scratch.lengths[static_cast<size_t>(idx)] == 0) {
      continue;
    }
    const std::string_view symbol =
        escaped.substr(ctx.scratch.offsets[static_cast<size_t>(idx)],
                       ctx.scratch.lengths[static_cast<size_t>(idx)]);
    const int32_t token = emel::encoder::detail::lookup_token(ctx, symbol);
    if (token != k_token_null) {
      if (!emel::encoder::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      continue;
    }
    for (const unsigned char c : symbol) {
      const int32_t byte_token =
          emel::encoder::detail::byte_to_token(ctx, vocab, c,
                                               emel::model::data::tokenizer_model::SPM);
      if (byte_token == k_token_null || !emel::encoder::detail::push_token(ev, byte_token, count)) {
        result.error = EMEL_ERR_BACKEND;
        return result;
      }
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::spm::detail
