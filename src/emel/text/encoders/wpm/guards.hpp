#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/text/encoders/wpm/context.hpp"
#include "emel/text/encoders/wpm/errors.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::wpm::guard {

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

struct table_sync_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::ok);
  }
};

struct table_sync_invalid_argument_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::invalid_argument);
  }
};

struct table_sync_backend_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::backend);
  }
};

struct table_sync_model_invalid_error {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return phase_error_is(ev, error::code::model_invalid);
  }
};

struct table_sync_unclassified_error_code {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    const int32_t err = ev.ctx.err;
    return err != error::to_emel(error::code::ok) &&
           err != error::to_emel(error::code::invalid_argument) &&
           err != error::to_emel(error::code::backend) &&
           err != error::to_emel(error::code::model_invalid);
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

struct encode_result_unclassified_error_code {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    const int32_t err = ev.ctx.err;
    return err != error::to_emel(error::code::ok) &&
           err != error::to_emel(error::code::invalid_argument) &&
           err != error::to_emel(error::code::backend) &&
           err != error::to_emel(error::code::model_invalid);
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

struct prefix_buffer_capacity_within_limit {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    constexpr size_t k_wpm_max_prefix_len = 3u;
    const bool has_prefix_capacity =
        ctx.scratch.buffer_alt.size() >= k_wpm_max_prefix_len;
    const size_t max_word_bytes =
      ctx.scratch.buffer_alt.size() -
      (k_wpm_max_prefix_len * static_cast<size_t>(has_prefix_capacity));
    return has_prefix_capacity && ev.request.text.size() <= max_word_bytes;
  }
};

struct prefix_buffer_capacity_exceeded {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !prefix_buffer_capacity_within_limit{}(ev, ctx);
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

struct tables_ready {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    (void)ev;
    return ctx.tables_ready && ctx.vocab == &ev.request.vocab;
  }
};

struct tables_missing {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !tables_ready{}(ev, ctx);
  }
};

}  // namespace emel::text::encoders::wpm::guard
