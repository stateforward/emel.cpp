#pragma once

#include "emel/text/encoders/plamo2/context.hpp"
#include "emel/text/encoders/plamo2/errors.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::plamo2::guard {

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

struct decode_result_empty_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok) && ev.data_len == 0;
  }
};

struct decode_result_non_empty_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok) && ev.data_len > 0;
  }
};

struct decode_result_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct decode_result_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct decode_result_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct decode_result_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !decode_result_empty_ok{}(ev) &&
           !decode_result_non_empty_ok{}(ev) &&
           !decode_result_invalid_argument_error{}(ev) &&
           !decode_result_backend_error{}(ev) &&
           !decode_result_model_invalid_error{}(ev);
  }
};

struct encode_result_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct encode_result_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct encode_result_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct encode_result_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct encode_result_unknown_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !encode_result_ok{}(ev) &&
           !encode_result_invalid_argument_error{}(ev) &&
           !encode_result_backend_error{}(ev) &&
           !encode_result_model_invalid_error{}(ev);
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
    return ctx.plamo2_tables_ready && ctx.plamo2_vocab == ctx.vocab;
  }
};

struct tables_missing {
  bool operator()(const runtime::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !tables_ready{}(ev, ctx);
  }
};

struct emit_result_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return ev.emit_result_error ==
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct emit_result_failed {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return !emit_result_ok{}(ev);
  }
};

}  // namespace emel::text::encoders::plamo2::guard
