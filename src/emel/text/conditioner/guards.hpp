#pragma once

#include "emel/text/conditioner/context.hpp"
#include "emel/text/conditioner/detail.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/conditioner/events.hpp"

namespace emel::text::conditioner::guard {

inline constexpr int32_t k_none_code = detail::to_local_error_code(error::none);
inline constexpr int32_t k_invalid_argument_code =
    detail::to_local_error_code(error::invalid_argument);
inline constexpr int32_t k_model_invalid_code =
    detail::to_local_error_code(error::model_invalid);
inline constexpr int32_t k_capacity_code =
    detail::to_local_error_code(error::capacity);
inline constexpr int32_t k_external_model_invalid_code =
    5; // legacy EMEL_ERR_MODEL_INVALID
inline constexpr int32_t k_external_backend_code = 6; // legacy EMEL_ERR_BACKEND
inline constexpr int32_t k_external_capacity_code =
    8; // legacy EMEL_ERR_CAPACITY

struct valid_bind {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    if (ev.request.tokenizer_sm == nullptr ||
        ev.request.dispatch_tokenizer_bind == nullptr ||
        ev.request.dispatch_tokenizer_tokenize == nullptr ||
        ev.request.format_prompt == nullptr) {
      return false;
    }

    switch (ev.request.preprocessor_variant) {
    case emel::text::tokenizer::preprocessor::preprocessor_kind::spm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::bpe:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::wpm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::ugm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::plamo2:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::fallback:
      break;
    default:
      return false;
    }

    switch (ev.request.encoder_variant) {
    case emel::text::encoders::encoder_kind::spm:
    case emel::text::encoders::encoder_kind::bpe:
    case emel::text::encoders::encoder_kind::wpm:
    case emel::text::encoders::encoder_kind::ugm:
    case emel::text::encoders::encoder_kind::rwkv:
    case emel::text::encoders::encoder_kind::plamo2:
    case emel::text::encoders::encoder_kind::fallback:
      break;
    default:
      return false;
    }

    return true;
  }
};

struct invalid_bind {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !valid_bind{}(runtime_ev);
  }
};

struct bind_rejected_no_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return !ev.ctx.bind_accepted && ev.ctx.bind_err_code == k_none_code;
  }
};

struct bind_error_code_present {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_err_code != k_none_code;
  }
};

struct bind_error_invalid_argument_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_err_code == k_invalid_argument_code;
  }
};

struct bind_error_model_invalid_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_err_code == k_model_invalid_code ||
           ev.ctx.bind_err_code == k_external_model_invalid_code;
  }
};

struct bind_error_capacity_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_err_code == k_capacity_code ||
           ev.ctx.bind_err_code == k_external_capacity_code;
  }
};

struct bind_error_backend_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_err_code == k_external_backend_code;
  }
};

struct bind_error_untracked_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return bind_error_code_present{}(runtime_ev) &&
           !bind_error_invalid_argument_code{}(runtime_ev) &&
           !bind_error_model_invalid_code{}(runtime_ev) &&
           !bind_error_capacity_code{}(runtime_ev) &&
           !bind_error_backend_code{}(runtime_ev);
  }
};

struct bind_successful {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.bind_accepted && ev.ctx.bind_err_code == k_none_code;
  }
};

struct has_bind_error_out {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.error_out != nullptr;
  }
};

struct no_bind_error_out {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_bind_error_out{}(runtime_ev);
  }
};

struct has_bind_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.dispatch_done != nullptr &&
           ev.request.owner_sm != nullptr;
  }
};

struct no_bind_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_bind_done_callback{}(runtime_ev);
  }
};

struct has_bind_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.dispatch_error != nullptr &&
           ev.request.owner_sm != nullptr;
  }
};

struct no_bind_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_bind_error_callback{}(runtime_ev);
  }
};

struct valid_prepare {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.is_bound && ctx.vocab != nullptr &&
           ctx.tokenizer_sm != nullptr &&
           ctx.dispatch_tokenizer_tokenize != nullptr &&
           ctx.format_prompt != nullptr &&
           ev.request.token_ids_out != nullptr &&
           ev.request.token_capacity > 0 && ev.ctx.formatted != nullptr &&
           ev.ctx.formatted_capacity > 0;
  }
};

struct invalid_prepare {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !valid_prepare{}(runtime_ev, ctx);
  }
};

struct use_bind_defaults {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.use_bind_defaults;
  }
};

struct use_request_overrides {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !use_bind_defaults{}(runtime_ev);
  }
};

struct valid_prepare_with_bind_defaults {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return valid_prepare{}(runtime_ev, ctx) && use_bind_defaults{}(runtime_ev);
  }
};

struct valid_prepare_with_request_overrides {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return valid_prepare{}(runtime_ev, ctx) &&
           use_request_overrides{}(runtime_ev);
  }
};

struct format_rejected_no_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return !ev.ctx.format_accepted && ev.ctx.format_err_code == k_none_code;
  }
};

struct format_error_code_present {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_err_code != k_none_code;
  }
};

struct format_error_invalid_argument_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_err_code == k_invalid_argument_code;
  }
};

struct format_error_model_invalid_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_err_code == k_model_invalid_code ||
           ev.ctx.format_err_code == k_external_model_invalid_code;
  }
};

struct format_error_capacity_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_err_code == k_capacity_code ||
           ev.ctx.format_err_code == k_external_capacity_code;
  }
};

struct format_error_backend_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_err_code == k_external_backend_code;
  }
};

struct format_error_untracked_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return format_error_code_present{}(runtime_ev) &&
           !format_error_invalid_argument_code{}(runtime_ev) &&
           !format_error_model_invalid_code{}(runtime_ev) &&
           !format_error_capacity_code{}(runtime_ev) &&
           !format_error_backend_code{}(runtime_ev);
  }
};

struct format_length_overflow {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_accepted && ev.ctx.format_err_code == k_none_code &&
           ev.ctx.formatted_length > ev.ctx.formatted_capacity;
  }
};

struct format_successful {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.format_accepted && ev.ctx.format_err_code == k_none_code &&
           ev.ctx.formatted_length <= ev.ctx.formatted_capacity;
  }
};

struct tokenize_rejected_no_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return !ev.ctx.tokenize_accepted && ev.ctx.tokenize_err_code == k_none_code;
  }
};

struct tokenize_error_code_present {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_err_code != k_none_code;
  }
};

struct tokenize_error_invalid_argument_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_err_code == k_invalid_argument_code;
  }
};

struct tokenize_error_model_invalid_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_err_code == k_model_invalid_code ||
           ev.ctx.tokenize_err_code == k_external_model_invalid_code;
  }
};

struct tokenize_error_capacity_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_err_code == k_capacity_code ||
           ev.ctx.tokenize_err_code == k_external_capacity_code;
  }
};

struct tokenize_error_backend_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_err_code == k_external_backend_code;
  }
};

struct tokenize_error_untracked_code {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return tokenize_error_code_present{}(runtime_ev) &&
           !tokenize_error_invalid_argument_code{}(runtime_ev) &&
           !tokenize_error_model_invalid_code{}(runtime_ev) &&
           !tokenize_error_capacity_code{}(runtime_ev) &&
           !tokenize_error_backend_code{}(runtime_ev);
  }
};

struct tokenize_count_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_accepted &&
           ev.ctx.tokenize_err_code == k_none_code &&
           (ev.ctx.token_count < 0 ||
            ev.ctx.token_count > ev.request.token_capacity);
  }
};

struct tokenize_successful {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.tokenize_accepted &&
           ev.ctx.tokenize_err_code == k_none_code && ev.ctx.token_count >= 0 &&
           ev.ctx.token_count <= ev.request.token_capacity;
  }
};

struct has_prepare_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.dispatch_done != nullptr &&
           ev.request.owner_sm != nullptr;
  }
};

struct no_prepare_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_prepare_done_callback{}(runtime_ev);
  }
};

struct has_prepare_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.dispatch_error != nullptr &&
           ev.request.owner_sm != nullptr;
  }
};

struct no_prepare_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev) const noexcept {
    return !has_prepare_error_callback{}(runtime_ev);
  }
};

} // namespace emel::text::conditioner::guard
