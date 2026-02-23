#pragma once

#include <cstring>
#include <limits>

#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::bpe::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

inline bool encode_bpe_word(const event::encode &ev,
                            emel::text::encoders::bpe::action::context &ctx,
                            const emel::model::data::vocab &vocab,
                            const std::string_view word,
                            int32_t &count,
                            encode_result &result) {
  if (word.empty()) {
    return true;
  }
  if (vocab.ignore_merges) {
    const int32_t token = emel::text::encoders::detail::lookup_token(ctx, word);
    if (token != k_token_null) {
      if (!emel::text::encoders::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
      }
      return true;
    }
  }

  if (!emel::text::encoders::detail::build_symbols(word, ctx.scratch, result)) {
    return false;
  }

  for (;;) {
    int32_t best_left = -1;
    int32_t best_right = -1;
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    for (int32_t left = 0; left != -1;
         left = ctx.scratch.next[static_cast<size_t>(left)]) {
      const int32_t right = ctx.scratch.next[static_cast<size_t>(left)];
      if (right < 0) {
        break;
      }
      const size_t left_off = ctx.scratch.offsets[static_cast<size_t>(left)];
      const size_t left_len = ctx.scratch.lengths[static_cast<size_t>(left)];
      const size_t right_off = ctx.scratch.offsets[static_cast<size_t>(right)];
      const size_t right_len = ctx.scratch.lengths[static_cast<size_t>(right)];
      const std::string_view left_view(word.data() + left_off, left_len);
      const std::string_view right_view(word.data() + right_off, right_len);
      const int32_t rank =
          emel::text::encoders::detail::lookup_merge_rank(ctx, vocab, left_view, right_view);
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
    emel::text::encoders::detail::merge_symbols(ctx.scratch, best_left, best_right);
  }

  for (int32_t idx = 0; idx != -1;
       idx = ctx.scratch.next[static_cast<size_t>(idx)]) {
    if (ctx.scratch.lengths[static_cast<size_t>(idx)] == 0) {
      continue;
    }
    const size_t sym_off = ctx.scratch.offsets[static_cast<size_t>(idx)];
    const size_t sym_len = ctx.scratch.lengths[static_cast<size_t>(idx)];
    const std::string_view symbol(word.data() + sym_off, sym_len);
    const int32_t token = emel::text::encoders::detail::lookup_token(ctx, symbol);
    if (token != k_token_null) {
      if (!emel::text::encoders::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
      }
      continue;
    }
    size_t byte_offset = 0;
    while (byte_offset < symbol.size()) {
      size_t len = emel::text::unicode_len_utf8(symbol[byte_offset]);
      if (byte_offset + len > symbol.size()) {
        len = 1;
      }
      const std::string_view unit(symbol.data() + byte_offset, len);
      const int32_t byte_token = emel::text::encoders::detail::lookup_token(ctx, unit);
      if (byte_token != k_token_null) {
        if (!emel::text::encoders::detail::push_token(ev, byte_token, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return false;
        }
      }
      byte_offset += len;
    }
  }

  return true;
}

inline encode_result encode_bpe(const event::encode &ev,
                                emel::text::encoders::bpe::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::text::encoders::detail::ensure_tables(ctx);

  int32_t count = 0;
  if (!ev.preprocessed) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }
  if (!encode_bpe_word(ev, ctx, vocab, ev.text, count, result)) {
    return result;
  }
  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::text::encoders::bpe::detail
