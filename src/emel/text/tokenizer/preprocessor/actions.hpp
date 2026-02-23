#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "emel/emel.h"
#include "emel/text/tokenizer/bpe/split.hpp"
#include "emel/text/tokenizer/preprocessor/context.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"

namespace emel::text::tokenizer::preprocessor::action {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void clear_request(context & ctx) noexcept {
  ctx.vocab = nullptr;
  ctx.text = {};
  ctx.fragments_out = nullptr;
  ctx.parse_special = false;
  ctx.preprocessed = false;
  ctx.fragment_capacity = 0;
  ctx.fragment_count = 0;
  ctx.bpe_scratch.reset();
}

struct begin_preprocess {
  void operator()(const event::preprocess & ev, context & ctx) const noexcept {
    if (ev.fragment_count_out != nullptr) {
      *ev.fragment_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.vocab = ev.vocab;
    ctx.text = ev.text;
    ctx.fragments_out = ev.fragments_out;
    ctx.parse_special = ev.parse_special;
    ctx.preprocessed = false;
    ctx.fragment_capacity = ev.fragment_capacity;
    ctx.fragment_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.bpe_scratch.reset();
    if (ev.preprocessed_out != nullptr) {
      *ev.preprocessed_out = false;
    }
  }
};

struct reject_invalid {
  void operator()(const event::preprocess &, context & ctx) const noexcept {
    ctx.fragment_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct build_specials {
  void operator()(context & ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.vocab == nullptr ||
        !detail::build_special_tokens(ctx.special_cache, *ctx.vocab)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_non_bpe {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.fragments_out == nullptr || ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (!detail::partition_with_specials(
            ctx.text, ctx.special_cache, ctx.parse_special,
            ctx.fragments_out, ctx.fragment_capacity,
            &ctx.fragment_count)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      ctx.preprocessed = false;
      return;
    }
    ctx.preprocessed = true;
  }
};

struct partition_bpe_no_specials {
  void operator()(context & ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.fragments_out == nullptr || ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    size_t out_count = 0;
    ctx.bpe_scratch.reset();
    emel::text::tokenizer::bpe::detail::split_view view = {};
    if (!emel::text::tokenizer::bpe::detail::split_and_encode_append(
            ctx.text, *ctx.vocab, ctx.bpe_scratch, view)) {
      ctx.fragment_count = 0;
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    for (size_t idx = 0; idx < view.count; ++idx) {
      const std::string_view word = view.words[idx];
      if (word.empty()) {
        continue;
      }
      if (!detail::push_raw_fragment(ctx.fragments_out,
                                     ctx.fragment_capacity, out_count,
                                     word)) {
        ctx.fragment_count = 0;
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
    }

    ctx.fragment_count = out_count;
    ctx.preprocessed = true;
  }
};

struct partition_bpe_with_specials {
  void operator()(context & ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.fragments_out == nullptr || ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    std::array<fragment, k_max_fragments> partitions = {};
    size_t partition_count = 0;
    if (!detail::partition_with_specials(ctx.text, ctx.special_cache,
                                         ctx.parse_special, partitions.data(),
                                         ctx.fragment_capacity,
                                         &partition_count)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    ctx.bpe_scratch.reset();
    size_t out_count = 0;
    for (size_t idx = 0; idx < partition_count; ++idx) {
      const fragment & frag = partitions[idx];
      if (frag.kind == fragment_kind::token) {
        if (!detail::push_token_fragment(ctx.fragments_out,
                                         ctx.fragment_capacity, out_count,
                                         frag.token)) {
          ctx.fragment_count = 0;
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
        continue;
      }

      if (frag.text.empty()) {
        continue;
      }
      emel::text::tokenizer::bpe::detail::split_view view = {};
      if (!emel::text::tokenizer::bpe::detail::split_and_encode_append(
              frag.text, *ctx.vocab, ctx.bpe_scratch, view)) {
        ctx.fragment_count = 0;
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
      for (size_t word_idx = 0; word_idx < view.count; ++word_idx) {
        const std::string_view word = view.words[word_idx];
        if (word.empty()) {
          continue;
        }
        if (!detail::push_raw_fragment(ctx.fragments_out,
                                       ctx.fragment_capacity, out_count,
                                       word)) {
          ctx.fragment_count = 0;
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
      }
    }

    ctx.fragment_count = out_count;
    ctx.preprocessed = true;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event &, context & ctx) const noexcept {
    ctx.fragment_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_preprocess begin_preprocess{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr build_specials build_specials{};
inline constexpr partition_non_bpe partition_non_bpe{};
inline constexpr partition_bpe_no_specials partition_bpe_no_specials{};
inline constexpr partition_bpe_with_specials partition_bpe_with_specials{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::tokenizer::preprocessor::action
