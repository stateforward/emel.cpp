#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

#include "emel/text/tokenizer/preprocessor/actions.hpp"

namespace emel::text::tokenizer::preprocessor::bpe::action {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

using emel::text::tokenizer::preprocessor::action::begin_preprocess;
using emel::text::tokenizer::preprocessor::action::build_specials;
using emel::text::tokenizer::preprocessor::action::clear_request;
using emel::text::tokenizer::preprocessor::action::context;
using emel::text::tokenizer::preprocessor::action::ensure_last_error;
using emel::text::tokenizer::preprocessor::action::mark_done;
using emel::text::tokenizer::preprocessor::action::on_unexpected;
using emel::text::tokenizer::preprocessor::action::reject_invalid;

namespace detail {

inline bool append_split_words_to_fragments(
  const event::preprocess & request,
  const emel::text::tokenizer::bpe::detail::split_view & view,
  size_t & out_count) {
  for (size_t idx = 0; idx < view.count; ++idx) {
    const std::string_view word = view.words[idx];
    if (!word.empty()) {
      if (!pdetail::push_raw_fragment(request.fragments_out.data(),
                                      request.fragments_out.size(), out_count, word)) {
        return false;
      }
    }
  }

  return true;
}

inline bool partition_bpe_no_specials(
  const event::preprocess & request,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  size_t & fragment_count_out) {
  fragment_count_out = 0;
  scratch.reset();

  emel::text::tokenizer::bpe::detail::split_view view = {};
  if (!emel::text::tokenizer::bpe::detail::split_and_encode_append(
        request.text, request.vocab, scratch, view)) {
    return false;
  }

  size_t out_count = 0;
  if (!append_split_words_to_fragments(request, view, out_count)) {
    return false;
  }

  fragment_count_out = out_count;
  return true;
}

inline bool append_partition_token_fragment(
  const event::preprocess & request,
  emel::text::tokenizer::bpe::detail::split_scratch &,
  const fragment & frag,
  size_t & out_count) {
  return pdetail::push_token_fragment(request.fragments_out.data(),
                                      request.fragments_out.size(), out_count,
                                      frag.token);
}

inline bool append_partition_raw_fragment(
  const event::preprocess & request,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  const fragment & frag,
  size_t & out_count) {
  if (frag.text.empty()) {
    return true;
  }
  emel::text::tokenizer::bpe::detail::split_view view = {};
  if (!emel::text::tokenizer::bpe::detail::split_and_encode_append(
        frag.text, request.vocab, scratch, view)) {
    return false;
  }
  return append_split_words_to_fragments(request, view, out_count);
}

inline bool append_partition_fragment(
  const event::preprocess & request,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  const fragment & frag,
  size_t & out_count) {
  using append_fn_type = bool (*)(const event::preprocess &,
                                  emel::text::tokenizer::bpe::detail::split_scratch &,
                                  const fragment &,
                                  size_t &);
  const std::array<append_fn_type, 2> appenders = {
    append_partition_raw_fragment,
    append_partition_token_fragment,
  };
  const size_t is_token = static_cast<size_t>(frag.kind == fragment_kind::token);
  return appenders[is_token](request, scratch, frag, out_count);
}

using special_partition_fn = bool (*)(std::string_view,
                                      const special_token_cache &,
                                      std::span<fragment>,
                                      size_t &);

inline bool partition_bpe_with_specials(
  const event::preprocess & request,
  const special_token_cache & cache,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  size_t & fragment_count_out,
  const special_partition_fn partition_specials) {
  fragment_count_out = 0;

  std::array<fragment, k_max_fragments> partitions = {};
  size_t partition_count = 0;
  if (!partition_specials(request.text, cache,
                          std::span<fragment>(partitions.data(),
                                              request.fragments_out.size()),
                          partition_count)) {
    return false;
  }

  scratch.reset();
  size_t out_count = 0;
  for (size_t idx = 0; idx < partition_count; ++idx) {
    const fragment & frag = partitions[idx];
    if (!append_partition_fragment(request, scratch, frag, out_count)) {
      return false;
    }
  }

  fragment_count_out = out_count;
  return true;
}

}  // namespace detail

struct partition_bpe_no_specials {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    const bool ok =
      detail::partition_bpe_no_specials(ev.request, ctx.bpe_scratch, fragment_count);
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, ok, fragment_count, true);
  }
};

struct partition_bpe_with_specials_parse_special {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    const bool ok = detail::partition_bpe_with_specials(
      ev.request, ctx.special_cache, ctx.bpe_scratch, fragment_count,
      pdetail::partition_with_specials_parse_enabled);
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, ok, fragment_count, true);
  }
};

struct partition_bpe_with_specials_skip_special {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    const bool ok = detail::partition_bpe_with_specials(
      ev.request, ctx.special_cache, ctx.bpe_scratch, fragment_count,
      pdetail::partition_with_specials_parse_disabled);
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, ok, fragment_count, true);
  }
};

inline constexpr partition_bpe_no_specials partition_bpe_no_specials{};
inline constexpr partition_bpe_with_specials_parse_special
  partition_bpe_with_specials_parse_special{};
inline constexpr partition_bpe_with_specials_skip_special
  partition_bpe_with_specials_skip_special{};

}  // namespace emel::text::tokenizer::preprocessor::bpe::action
