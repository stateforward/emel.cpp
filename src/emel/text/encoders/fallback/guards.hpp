#pragma once

#include "emel/text/encoders/fallback/context.hpp"
#include "emel/text/encoders/fallback/errors.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::fallback::guard {

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

struct table_prepare_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct table_prepare_invalid_argument_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct table_prepare_backend_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct table_prepare_model_invalid_error {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct table_prepare_unclassified_error_code {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    const auto err = ev.event_.ctx.err;
    return err != error::to_emel(error::code::ok) &&
           err != error::to_emel(error::code::invalid_argument) &&
           err != error::to_emel(error::code::backend) &&
           err != error::to_emel(error::code::model_invalid);
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

struct encode_result_unclassified_error_code {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    const auto err = ev.event_.ctx.err;
    return err != error::to_emel(error::code::ok) &&
           err != error::to_emel(error::code::invalid_argument) &&
           err != error::to_emel(error::code::backend) &&
           err != error::to_emel(error::code::model_invalid);
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

}  // namespace emel::text::encoders::fallback::guard
