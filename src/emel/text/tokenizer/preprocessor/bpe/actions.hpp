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

inline bool append_split_words_noop(
  const event::preprocess &,
  const emel::text::tokenizer::bpe::detail::split_view &,
  size_t &) {
  return true;
}

inline bool append_split_words_to_fragments(
  const event::preprocess & request,
  const emel::text::tokenizer::bpe::detail::split_view & view,
  size_t & out_count) {
  bool ok = true;
  for (size_t idx = 0; idx < view.count; ++idx) {
    const std::string_view word = view.words[idx];
    const bool step_active = ok;
    const size_t emit_idx = static_cast<size_t>(step_active && !word.empty());
    const std::array<std::string_view, 2> emitted_words{
      std::string_view{},
      word,
    };
    const bool push_ok =
      pdetail::push_raw_fragment(request.fragments_out.data(),
                                 request.fragments_out.size(), out_count,
                                 emitted_words[emit_idx]);
    ok = ok && push_ok;
  }

  return ok;
}

inline bool split_words_noop(
  std::string_view,
  const emel::model::data::vocab &,
  emel::text::tokenizer::bpe::detail::split_scratch &,
  emel::text::tokenizer::bpe::detail::split_view &) {
  return true;
}

inline bool split_words_encoded(
  const std::string_view text,
  const emel::model::data::vocab & vocab,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  emel::text::tokenizer::bpe::detail::split_view & view) {
  return emel::text::tokenizer::bpe::detail::split_and_encode_append(
    text, vocab, scratch, view);
}

inline bool append_partition_fragment_noop(
  const event::preprocess &,
  emel::text::tokenizer::bpe::detail::split_scratch &,
  const fragment &,
  size_t &) {
  return true;
}

inline void reset_split_scratch_noop(
  emel::text::tokenizer::bpe::detail::split_scratch &) {}

inline void reset_split_scratch_active(
  emel::text::tokenizer::bpe::detail::split_scratch & scratch) {
  scratch.reset();
}

inline bool partition_bpe_no_specials(
  const event::preprocess & request,
  emel::text::tokenizer::bpe::detail::split_scratch & scratch,
  size_t & fragment_count_out) {
  fragment_count_out = 0;
  scratch.reset();

  emel::text::tokenizer::bpe::detail::split_view view = {};
  size_t out_count = 0;
  const bool split_ok = emel::text::tokenizer::bpe::detail::split_and_encode_append(
    request.text, request.vocab, scratch, view);
  using append_words_fn =
    bool (*)(const event::preprocess &,
             const emel::text::tokenizer::bpe::detail::split_view &,
             size_t &);
  const std::array<append_words_fn, 2> appenders{
    append_split_words_noop,
    append_split_words_to_fragments,
  };
  const bool append_ok = appenders[static_cast<size_t>(split_ok)](
    request, view, out_count);
  const bool ok = split_ok && append_ok;
  const std::array<size_t, 2> counts{0, out_count};
  fragment_count_out = counts[static_cast<size_t>(ok)];
  return ok;
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
  const bool has_text = !frag.text.empty();
  using split_words_fn =
    bool (*)(std::string_view,
             const emel::model::data::vocab &,
             emel::text::tokenizer::bpe::detail::split_scratch &,
             emel::text::tokenizer::bpe::detail::split_view &);
  const std::array<split_words_fn, 2> splitters{
    split_words_noop,
    split_words_encoded,
  };

  emel::text::tokenizer::bpe::detail::split_view view = {};
  const bool split_ok = splitters[static_cast<size_t>(has_text)](
    frag.text, request.vocab, scratch, view);

  using append_words_fn =
    bool (*)(const event::preprocess &,
             const emel::text::tokenizer::bpe::detail::split_view &,
             size_t &);
  const std::array<append_words_fn, 2> appenders{
    append_split_words_noop,
    append_split_words_to_fragments,
  };
  const size_t append_idx = static_cast<size_t>(has_text && split_ok);
  const bool append_ok = appenders[append_idx](request, view, out_count);
  const std::array<bool, 2> results{true, split_ok && append_ok};
  return results[static_cast<size_t>(has_text)];
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
  bool ok = partition_specials(request.text, cache,
                               std::span<fragment>(partitions.data(),
                                                   request.fragments_out.size()),
                               partition_count);

  using reset_scratch_fn =
    void (*)(emel::text::tokenizer::bpe::detail::split_scratch &);
  const std::array<reset_scratch_fn, 2> resetters{
    reset_split_scratch_noop,
    reset_split_scratch_active,
  };
  resetters[static_cast<size_t>(ok)](scratch);

  size_t out_count = 0;
  using append_fn_type = bool (*)(const event::preprocess &,
                                  emel::text::tokenizer::bpe::detail::split_scratch &,
                                  const fragment &,
                                  size_t &);
  const std::array<append_fn_type, 2> appenders = {
    append_partition_fragment_noop,
    append_partition_fragment,
  };

  for (size_t idx = 0; idx < partition_count; ++idx) {
    const bool step_active = ok;
    const fragment & frag = partitions[idx];
    const bool step_ok = appenders[static_cast<size_t>(step_active)](
      request, scratch, frag, out_count);
    ok = ok && step_ok;
  }

  const std::array<size_t, 2> counts{0, out_count};
  fragment_count_out = counts[static_cast<size_t>(ok)];
  return ok;
}

}  // namespace detail

struct set_empty_partition_result {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context &) const noexcept {
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, true, 0, true);
  }
};

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

inline constexpr set_empty_partition_result set_empty_partition_result{};
inline constexpr partition_bpe_no_specials partition_bpe_no_specials{};
inline constexpr partition_bpe_with_specials_parse_special
  partition_bpe_with_specials_parse_special{};
inline constexpr partition_bpe_with_specials_skip_special
  partition_bpe_with_specials_skip_special{};

}  // namespace emel::text::tokenizer::preprocessor::bpe::action
