#pragma once

#include <cstring>
#include <limits>

#include "emel/encoder/bpe/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"
#include "emel/tokenizer/bpe/split.hpp"

namespace emel::encoder::bpe::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline bool encode_bpe_word(const event::encode &ev,
                            emel::encoder::bpe::action::context &ctx,
                            const emel::model::data::vocab &vocab,
                            const std::string_view word,
                            int32_t &count,
                            encode_result &result) {
  if (word.empty()) {
    return true;
  }
  if (vocab.ignore_merges) {
    const int32_t token = emel::encoder::detail::lookup_token(ctx, word);
    if (token != k_token_null) {
      if (!emel::encoder::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
      }
      return true;
    }
  }

  if (!emel::encoder::detail::build_symbols(word, ctx.scratch, result)) {
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

  for (int32_t idx = 0; idx != -1;
       idx = ctx.scratch.next[static_cast<size_t>(idx)]) {
    if (ctx.scratch.lengths[static_cast<size_t>(idx)] == 0) {
      continue;
    }
    const size_t sym_off = ctx.scratch.offsets[static_cast<size_t>(idx)];
    const size_t sym_len = ctx.scratch.lengths[static_cast<size_t>(idx)];
    const std::string_view symbol(word.data() + sym_off, sym_len);
    const int32_t token = emel::encoder::detail::lookup_token(ctx, symbol);
    if (token != k_token_null) {
      if (!emel::encoder::detail::push_token(ev, token, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
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
          return false;
        }
      }
    }
  }

  return true;
}

inline encode_result encode_bpe(const event::encode &ev,
                                emel::encoder::bpe::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  int32_t count = 0;
  if (ev.pretokenized) {
    if (!encode_bpe_word(ev, ctx, vocab, ev.text, count, result)) {
      return result;
    }
    result.token_count = count;
    result.error = EMEL_OK;
    return result;
  }

  ctx.bpe_scratch.reset();
  emel::tokenizer::bpe::detail::split_view view = {};
  if (!emel::tokenizer::bpe::detail::split_and_encode_append(
          ev.text, vocab, ctx.bpe_scratch, view)) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }
  for (size_t idx = 0; idx < view.count; ++idx) {
    if (!encode_bpe_word(ev, ctx, vocab, view.words[idx], count, result)) {
      return result;
    }
    if (result.error != EMEL_OK) {
      return result;
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::bpe::detail
