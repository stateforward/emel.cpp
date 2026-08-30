#pragma once

#include "emel/text/tokenizer/preprocessor/actions.hpp"

namespace emel::text::tokenizer::preprocessor::spm::action {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

using emel::text::tokenizer::preprocessor::action::begin_preprocess;
using emel::text::tokenizer::preprocessor::action::build_specials;
using emel::text::tokenizer::preprocessor::action::clear_request;
using emel::text::tokenizer::preprocessor::action::context;
using emel::text::tokenizer::preprocessor::action::ensure_last_error;
using emel::text::tokenizer::preprocessor::action::mark_done;
using emel::text::tokenizer::preprocessor::action::on_unexpected;
using emel::text::tokenizer::preprocessor::action::reject_invalid;

struct set_empty_partition_result {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context &) const noexcept {
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, true, 0, true);
  }
};

struct partition_no_specials {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    fragment & first = ev.request.fragments_out[0];
    first.kind = fragment_kind::raw_text;
    first.text = ev.request.text;
    first.token = -1;
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, true, 1, true);
  }
};

inline bool prepend_needle_dummy_prefix(
    const event::preprocess & request, size_t & fragment_count) noexcept {
  static constexpr char k_space = ' ';
  const bool needle_profile =
    request.vocab.tokenizer_pre_id == emel::model::data::tokenizer_pre::NEEDLE;
  const bool starts_with_special =
    fragment_count != 0u && request.fragments_out[0].kind == fragment_kind::token;
  if (!needle_profile || !request.vocab.add_space_prefix || !starts_with_special) {
    return true;
  }
  if (fragment_count >= request.fragments_out.size()) {
    return false;
  }
  for (size_t i = fragment_count; i > 0u; --i) {
    request.fragments_out[i] = request.fragments_out[i - 1u];
  }
  fragment & prefix = request.fragments_out[0];
  prefix.kind = fragment_kind::raw_text;
  prefix.text = std::string_view{&k_space, 1u};
  prefix.token = -1;
  ++fragment_count;
  return true;
}

struct partition_non_bpe_parse_special {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    const bool partitioned = pdetail::partition_with_specials_parse_enabled(
      ev.request.text, ctx.special_cache, ev.request.fragments_out, fragment_count);
    const bool ok = partitioned &&
                    prepend_needle_dummy_prefix(ev.request, fragment_count);
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, ok, fragment_count, true);
  }
};

struct partition_non_bpe_skip_special {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    const bool ok = pdetail::partition_with_specials_parse_disabled(
      ev.request.text, ctx.special_cache, ev.request.fragments_out, fragment_count);
    emel::text::tokenizer::preprocessor::action::detail::set_phase_result(
      runtime_ev, ok, fragment_count, true);
  }
};

inline constexpr set_empty_partition_result set_empty_partition_result{};
inline constexpr partition_no_specials partition_no_specials{};
inline constexpr partition_non_bpe_parse_special partition_non_bpe_parse_special{};
inline constexpr partition_non_bpe_skip_special partition_non_bpe_skip_special{};

}  // namespace emel::text::tokenizer::preprocessor::spm::action
