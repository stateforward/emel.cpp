#pragma once

#include <cstring>
#include <limits>

#include "emel/encoder/bpe/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"
#include "emel/tokenizer/bpe/regex.hpp"

namespace emel::encoder::bpe::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline void assign_bpe_regex(action::context &ctx,
                             const emel::model::data::vocab &vocab) {
  emel::tokenizer::bpe::detail::assign_bpe_regex(ctx.bpe_pre_id,
                                                 ctx.bpe_regex_exprs, vocab);
}

inline encode_result encode_bpe(const event::encode &ev,
                                emel::encoder::bpe::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  assign_bpe_regex(ctx, vocab);
  const std::string text(ev.text);
  const auto words = emel::text::unicode_regex_split(text, ctx.bpe_regex_exprs);
  int32_t count = 0;
  for (const std::string &word : words) {
    if (word.empty()) {
      continue;
    }
    if (vocab.ignore_merges) {
      const int32_t token = emel::encoder::detail::lookup_token(ctx, word);
      if (token != k_token_null) {
        if (!emel::encoder::detail::push_token(ev, token, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
        continue;
      }
    }

    if (!emel::encoder::detail::build_symbols(word, ctx.scratch, result)) {
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

    for (int32_t idx = 0; idx != -1; idx = ctx.scratch.next[static_cast<size_t>(idx)]) {
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
        }
      }
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::bpe::detail
