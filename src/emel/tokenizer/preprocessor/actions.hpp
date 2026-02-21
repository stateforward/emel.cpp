#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "emel/emel.h"
#include "emel/text/unicode.hpp"
#include "emel/tokenizer/bpe/regex.hpp"
#include "emel/tokenizer/preprocessor/context.hpp"
#include "emel/tokenizer/preprocessor/detail.hpp"

namespace emel::tokenizer::preprocessor::action {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void clear_request(context & ctx) noexcept {
  ctx.request = nullptr;
  ctx.vocab = nullptr;
  ctx.text = {};
  ctx.parse_special = false;
  ctx.fragment_capacity = 0;
  ctx.fragment_count = 0;
  ctx.bpe_words.clear();
}

struct begin_preprocess {
  void operator()(const event::preprocess & ev, context & ctx) const noexcept {
    if (ev.fragment_count_out != nullptr) {
      *ev.fragment_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.request = &ev;
    ctx.vocab = ev.vocab;
    ctx.text = ev.text;
    ctx.parse_special = ev.parse_special;
    ctx.fragment_capacity = ev.fragment_capacity;
    ctx.fragment_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.bpe_words.clear();
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
    if (ctx.request == nullptr || ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (!detail::partition_with_specials(
            ctx.text, ctx.special_cache, ctx.parse_special,
            ctx.request->fragments_out, ctx.fragment_capacity,
            &ctx.fragment_count)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct partition_bpe_no_specials {
  void operator()(context & ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.request == nullptr || ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    emel::tokenizer::bpe::detail::assign_bpe_regex(ctx.bpe_pre_id,
                                                   ctx.bpe_regex_exprs,
                                                   *ctx.vocab);
    ctx.bpe_words.clear();
    if (ctx.bpe_words.capacity() < ctx.fragment_capacity) {
      ctx.bpe_words.reserve(ctx.fragment_capacity);
    }

    size_t out_count = 0;
    const std::string raw_text(ctx.text);
    const auto words = emel::text::unicode_regex_split(raw_text,
                                                       ctx.bpe_regex_exprs);
    for (const std::string & word : words) {
      if (word.empty()) {
        continue;
      }
      if (out_count >= ctx.fragment_capacity) {
        ctx.fragment_count = 0;
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
      ctx.bpe_words.push_back(word);
      if (!detail::push_raw_fragment(ctx.request->fragments_out,
                                     ctx.fragment_capacity, out_count,
                                     ctx.bpe_words.back())) {
        ctx.fragment_count = 0;
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
    }

    ctx.fragment_count = out_count;
  }
};

struct partition_bpe_with_specials {
  void operator()(context & ctx) const {
    ctx.phase_error = EMEL_OK;
    if (ctx.request == nullptr || ctx.vocab == nullptr) {
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

    emel::tokenizer::bpe::detail::assign_bpe_regex(ctx.bpe_pre_id,
                                                   ctx.bpe_regex_exprs,
                                                   *ctx.vocab);
    ctx.bpe_words.clear();
    if (ctx.bpe_words.capacity() < ctx.fragment_capacity) {
      ctx.bpe_words.reserve(ctx.fragment_capacity);
    }

    size_t out_count = 0;
    for (size_t idx = 0; idx < partition_count; ++idx) {
      const fragment & frag = partitions[idx];
      if (frag.kind == fragment_kind::token) {
        if (!detail::push_token_fragment(ctx.request->fragments_out,
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

      const std::string raw_text(frag.text);
      const auto words = emel::text::unicode_regex_split(raw_text,
                                                         ctx.bpe_regex_exprs);
      for (const std::string & word : words) {
        if (word.empty()) {
          continue;
        }
        if (out_count >= ctx.fragment_capacity) {
          ctx.fragment_count = 0;
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
        ctx.bpe_words.push_back(word);
        if (!detail::push_raw_fragment(ctx.request->fragments_out,
                                       ctx.fragment_capacity, out_count,
                                       ctx.bpe_words.back())) {
          ctx.fragment_count = 0;
          set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
      }
    }

    ctx.fragment_count = out_count;
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

}  // namespace emel::tokenizer::preprocessor::action
