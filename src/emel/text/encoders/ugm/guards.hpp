#pragma once

#include "emel/text/encoders/ugm/context.hpp"
#include "emel/text/encoders/ugm/errors.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::ugm::guard {

inline bool phase_error_is(const runtime::encode_runtime & ev,
                           const error::code code_value) noexcept {
  return ev.event_.ctx.err == error::to_emel(code_value);
}

struct valid_encode {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode{}(ev.event_, ctx);
  }
};

struct invalid_encode {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::invalid_encode{}(ev.event_, ctx);
  }
};

struct table_sync_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct table_sync_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct table_sync_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct table_sync_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct table_sync_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !table_sync_ok{}(ev) &&
           !table_sync_invalid_argument_error{}(ev) &&
           !table_sync_backend_error{}(ev) &&
           !table_sync_model_invalid_error{}(ev);
  }
};

struct normalize_result_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct normalize_result_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct normalize_result_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct normalize_result_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct normalize_result_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !normalize_result_ok{}(ev) &&
           !normalize_result_invalid_argument_error{}(ev) &&
           !normalize_result_backend_error{}(ev) &&
           !normalize_result_model_invalid_error{}(ev);
  }
};

struct input_prepare_result_empty_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok) && ev.normalized.empty();
  }
};

struct input_prepare_result_non_empty_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok) && !ev.normalized.empty();
  }
};

struct input_prepare_result_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct input_prepare_result_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct input_prepare_result_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct input_prepare_result_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !input_prepare_result_empty_ok{}(ev) &&
           !input_prepare_result_non_empty_ok{}(ev) &&
           !input_prepare_result_invalid_argument_error{}(ev) &&
           !input_prepare_result_backend_error{}(ev) &&
           !input_prepare_result_model_invalid_error{}(ev);
  }
};

struct dp_forward_result_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct dp_forward_result_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct dp_forward_result_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct dp_forward_result_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct dp_forward_result_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !dp_forward_result_ok{}(ev) &&
           !dp_forward_result_invalid_argument_error{}(ev) &&
           !dp_forward_result_backend_error{}(ev) &&
           !dp_forward_result_model_invalid_error{}(ev);
  }
};

struct text_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_empty{}(ev.event_);
  }
};

struct text_non_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_non_empty{}(ev.event_);
  }
};

struct vocab_changed {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_changed{}(ev.event_, ctx);
  }
};

struct vocab_unchanged {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_unchanged{}(ev.event_, ctx);
  }
};

struct tables_ready {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    (void)ev;
    return ctx.ugm_tables_ready && ctx.ugm_vocab == ctx.vocab;
  }
};

struct tables_missing {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !tables_ready{}(ev, ctx);
  }
};

struct vocab_unk_present {
  bool operator()(const runtime::encode_runtime &, const action::context & ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->unk_id != emel::text::encoders::detail::k_token_null;
  }
};

struct vocab_unk_missing {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !vocab_unk_present{}(ev, ctx);
  }
};

struct backtrace_failed {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return ev.backtrace_failed;
  }
};

struct backtrace_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !ev.backtrace_failed;
  }
};

struct emit_failed {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return ev.emit_failed;
  }
};

struct emit_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !ev.emit_failed;
  }
};

}  // namespace emel::text::encoders::ugm::guard
