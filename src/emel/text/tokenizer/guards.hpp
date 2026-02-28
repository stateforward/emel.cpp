#pragma once

#include <cstdint>

#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/errors.hpp"

namespace emel::text::tokenizer::guard {

inline constexpr int32_t k_none_code = error_code(error::none);

struct can_tokenize {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    if (!ctx.is_bound || ctx.vocab == nullptr) {
      return false;
    }
    if (ev.vocab == nullptr || ev.vocab != ctx.vocab) {
      return false;
    }
    if (ev.token_ids_out == nullptr || ev.token_count_out == nullptr) {
      return false;
    }
    return ev.token_capacity > 0;
  }

  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return operator()(ev.request, ctx);
  }
};

struct can_bind {
  bool operator()(const event::bind &ev) const noexcept {
    if (ev.vocab == nullptr) {
      return false;
    }
    switch (ev.preprocessor_variant) {
    case action::preprocessor_kind::spm:
    case action::preprocessor_kind::bpe:
    case action::preprocessor_kind::wpm:
    case action::preprocessor_kind::ugm:
    case action::preprocessor_kind::rwkv:
    case action::preprocessor_kind::plamo2:
    case action::preprocessor_kind::fallback:
      break;
    default:
      return false;
    }
    switch (ev.encoder_variant) {
    case action::encoder_kind::spm:
    case action::encoder_kind::bpe:
    case action::encoder_kind::wpm:
    case action::encoder_kind::ugm:
    case action::encoder_kind::rwkv:
    case action::encoder_kind::plamo2:
    case action::encoder_kind::fallback:
      break;
    default:
      return false;
    }
    return true;
  }

  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return operator()(ev.request);
  }
};

struct phase_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.err == k_none_code;
  }
};

struct phase_failed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !phase_ok{}(runtime_ev);
  }
};

struct preprocess_rejected_no_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return !ev.ctx.preprocess_accepted &&
           ev.ctx.preprocess_err_code == k_none_code;
  }
};

struct preprocess_reported_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.preprocess_err_code != k_none_code;
  }
};

struct preprocess_fragment_count_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.preprocess_accepted &&
           ev.ctx.preprocess_err_code == k_none_code &&
           ev.ctx.fragment_count > ev.ctx.fragment_capacity;
  }
};

struct preprocess_success {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.preprocess_accepted &&
           ev.ctx.preprocess_err_code == k_none_code &&
           ev.ctx.fragment_count <= ev.ctx.fragment_capacity;
  }
};

struct has_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.token_count < ev.request.token_capacity;
  }
};

struct no_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_capacity{}(runtime_ev);
  }
};

struct should_add_bos {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.request.add_special && ctx.vocab != nullptr && ctx.vocab->add_bos;
  }
};

struct no_prefix {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !should_add_bos{}(runtime_ev, ctx);
  }
};

struct bos_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->bos_id >= 0;
  }
};

struct bos_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !bos_id_valid{}(ctx);
  }
};

struct bos_ready {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_bos{}(runtime_ev, ctx) && bos_id_valid{}(ctx) &&
           has_capacity{}(runtime_ev);
  }
};

struct bos_no_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_bos{}(runtime_ev, ctx) && bos_id_valid{}(ctx) &&
           no_capacity{}(runtime_ev);
  }
};

struct bos_invalid_id {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_bos{}(runtime_ev, ctx) && bos_id_invalid{}(ctx);
  }
};

struct should_add_sep {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.request.add_special && ctx.vocab != nullptr &&
           ctx.model_kind == action::encoder_kind::wpm && ctx.vocab->add_sep;
  }
};

struct should_add_eos {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.request.add_special && ctx.vocab != nullptr &&
           ctx.model_kind != action::encoder_kind::wpm && ctx.vocab->add_eos;
  }
};

struct no_suffix {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !should_add_sep{}(runtime_ev, ctx) &&
           !should_add_eos{}(runtime_ev, ctx);
  }
};

struct sep_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->sep_id >= 0;
  }
};

struct sep_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !sep_id_valid{}(ctx);
  }
};

struct sep_ready {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_sep{}(runtime_ev, ctx) && sep_id_valid{}(ctx) &&
           has_capacity{}(runtime_ev);
  }
};

struct sep_no_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_sep{}(runtime_ev, ctx) && sep_id_valid{}(ctx) &&
           no_capacity{}(runtime_ev);
  }
};

struct sep_invalid_id {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_sep{}(runtime_ev, ctx) && sep_id_invalid{}(ctx);
  }
};

struct eos_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->eos_id >= 0;
  }
};

struct eos_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !eos_id_valid{}(ctx);
  }
};

struct eos_ready {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_eos{}(runtime_ev, ctx) && eos_id_valid{}(ctx) &&
           has_capacity{}(runtime_ev);
  }
};

struct eos_no_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_eos{}(runtime_ev, ctx) && eos_id_valid{}(ctx) &&
           no_capacity{}(runtime_ev);
  }
};

struct eos_invalid_id {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return should_add_eos{}(runtime_ev, ctx) && eos_id_invalid{}(ctx);
  }
};

struct has_more_fragments {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.fragment_index < ev.ctx.fragment_count;
  }
};

struct no_more_fragments {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_more_fragments{}(runtime_ev);
  }
};

struct fragment_is_token {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return has_more_fragments{}(runtime_ev) &&
           ev.ctx.fragments[ev.ctx.fragment_index].kind ==
               action::fragment_kind::token;
  }
};

struct fragment_token_valid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return fragment_is_token{}(runtime_ev) &&
           ev.ctx.fragments[ev.ctx.fragment_index].token >= 0;
  }
};

struct fragment_token_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return fragment_is_token{}(runtime_ev) &&
           ev.ctx.fragments[ev.ctx.fragment_index].token < 0;
  }
};

struct fragment_is_raw {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return has_more_fragments{}(runtime_ev) &&
           ev.ctx.fragments[ev.ctx.fragment_index].kind ==
               action::fragment_kind::raw_text;
  }
};

struct more_fragments_no_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return has_more_fragments{}(runtime_ev) && no_capacity{}(runtime_ev);
  }
};

struct more_fragments_token_valid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return has_more_fragments{}(runtime_ev) &&
           fragment_token_valid{}(runtime_ev);
  }
};

struct more_fragments_token_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return has_more_fragments{}(runtime_ev) &&
           fragment_token_invalid{}(runtime_ev);
  }
};

struct more_fragments_raw {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return has_more_fragments{}(runtime_ev) && fragment_is_raw{}(runtime_ev);
  }
};

struct encode_rejected_no_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return !ev.ctx.encode_accepted && ev.ctx.encode_err_code == k_none_code;
  }
};

struct encode_reported_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.encode_err_code != k_none_code;
  }
};

struct encode_count_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    const int32_t remaining_capacity =
        ev.request.token_capacity - ev.ctx.token_count;
    return ev.ctx.encode_accepted && ev.ctx.encode_err_code == k_none_code &&
           (ev.ctx.encode_token_count < 0 ||
            ev.ctx.encode_token_count > remaining_capacity);
  }
};

struct encode_success {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.encode_accepted && ev.ctx.encode_err_code == k_none_code &&
           !encode_count_invalid{}(runtime_ev);
  }
};

} // namespace emel::text::tokenizer::guard
