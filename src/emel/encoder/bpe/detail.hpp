#pragma once

#include <cstring>
#include <limits>

#include "emel/encoder/bpe/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::bpe::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline void assign_bpe_regex(action::context &ctx,
                             const emel::model::data::vocab &vocab) {
  ctx.bpe_pre = std::string(emel::encoder::detail::string_view_from_array(vocab.tokenizer_pre));
  ctx.bpe_regex_exprs.clear();
  ctx.bpe_regex_exprs.emplace_back(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
}

inline encode_result encode_bpe(const event::encode &ev,
                                emel::encoder::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  size_t out_len = 0;
  for (const unsigned char c : ev.text) {
    const uint32_t cpt = emel::encoder::detail::byte_to_codepoint_table()[c];
    char utf8[4] = {};
    const uint8_t len = emel::encoder::detail::encode_cpt_utf8(cpt, utf8);
    if (out_len + len > ctx.scratch.buffer.size()) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
    std::memcpy(ctx.scratch.buffer.data() + out_len, utf8, len);
    out_len += len;
  }
  const std::string_view encoded(ctx.scratch.buffer.data(), out_len);

  if (vocab.ignore_merges) {
    const int32_t token = emel::encoder::detail::lookup_token(ctx, encoded);
    if (token != k_token_null) {
      int32_t count = 0;
      if (!emel::encoder::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      result.token_count = count;
      return result;
    }
  }

  if (!emel::encoder::detail::build_symbols(encoded, ctx.scratch, result)) {
    return result;
  }

  for (;;) {
    int32_t best_left = -1;
    int32_t best_right = -1;
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    for (int32_t left = 0; left != -1; left = ctx.scratch.next[static_cast<size_t>(left)]) {
      const int32_t right = ctx.scratch.next[static_cast<size_t>(left)];
      if (right < 0) {
        break;
      }
      const std::string_view left_view =
          encoded.substr(ctx.scratch.offsets[static_cast<size_t>(left)],
                         ctx.scratch.lengths[static_cast<size_t>(left)]);
      const std::string_view right_view =
          encoded.substr(ctx.scratch.offsets[static_cast<size_t>(right)],
                         ctx.scratch.lengths[static_cast<size_t>(right)]);
      const int32_t rank =
          emel::encoder::detail::lookup_merge_rank(ctx, vocab, left_view, right_view);
      if (rank == k_token_null) {
        continue;
      }
      if (rank < best_rank || (rank == best_rank && left < best_left)) {
        best_rank = rank;
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
        encoded.substr(ctx.scratch.offsets[static_cast<size_t>(idx)],
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
      const char byte = static_cast<char>(c);
      const int32_t byte_token =
          emel::encoder::detail::lookup_token(ctx, std::string_view(&byte, 1));
      if (byte_token != k_token_null) {
        if (!emel::encoder::detail::push_token(ev, byte_token, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
        continue;
      }
      int32_t unk = vocab.unk_id;
      if (unk == k_token_null) {
        unk = emel::encoder::detail::lookup_token(ctx, "<unk>");
      }
      if (unk == k_token_null) {
        continue;
      }
      if (!emel::encoder::detail::push_token(ev, unk, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::bpe::detail
