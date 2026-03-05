#pragma once

#include "emel/text/encoders/bpe/detail.hpp"
#include "emel/text/encoders/bpe/errors.hpp"
#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::bpe::guard {

inline bool phase_error_is(const event::encode_runtime & ev,
                           const error::code code_value) noexcept {
  return ev.ctx.err == error::to_emel(code_value);
}

struct valid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode{}(ev, ctx);
  }
};

struct invalid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::invalid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_ok{}(ev);
  }
};

struct phase_failed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_failed{}(ev);
  }
};

struct table_prepare_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct table_prepare_invalid_argument_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct table_prepare_backend_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct table_prepare_model_invalid_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct table_prepare_unknown_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return !table_prepare_ok{}(ev) &&
           !table_prepare_invalid_argument_error{}(ev) &&
           !table_prepare_backend_error{}(ev) &&
           !table_prepare_model_invalid_error{}(ev);
  }
};

struct encode_result_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct encode_result_invalid_argument_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct encode_result_backend_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct encode_result_model_invalid_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct encode_result_unknown_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return !encode_result_ok{}(ev) &&
           !encode_result_invalid_argument_error{}(ev) &&
           !encode_result_backend_error{}(ev) &&
           !encode_result_model_invalid_error{}(ev);
  }
};

struct text_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_empty{}(ev);
  }
};

struct text_non_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_non_empty{}(ev);
  }
};

struct preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::preprocessed{}(ev);
  }
};

struct not_preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::not_preprocessed{}(ev);
  }
};

struct ignore_merges_enabled {
  bool operator()(const event::encode_runtime &, const action::context & ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->ignore_merges;
  }
};

struct direct_word_token_available {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::bpe::detail::bpe_lookup_token(ctx, ev.request.text) !=
           emel::text::encoders::bpe::detail::k_token_null;
  }
};

struct merge_symbol_capacity_within_limit {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return ev.request.text.size() <= ctx.scratch.offsets.size();
  }
};

struct merge_symbol_capacity_exceeded {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !merge_symbol_capacity_within_limit{}(ev, ctx);
  }
};

struct vocab_changed {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_changed{}(ev, ctx);
  }
};

struct vocab_unchanged {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_unchanged{}(ev, ctx);
  }
};

}  // namespace emel::text::encoders::bpe::guard
