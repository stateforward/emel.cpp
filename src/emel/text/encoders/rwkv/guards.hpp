#pragma once

#include "emel/text/encoders/rwkv/context.hpp"
#include "emel/text/encoders/rwkv/errors.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::rwkv::guard {

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

struct table_sync_unclassified_error_code {
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

struct output_capacity_covers_text {
  bool operator()(const runtime::encode_runtime & ev,
                  const action::context &) const noexcept {
    return ev.event_.request.token_ids.size() >= ev.event_.request.text.size();
  }
};

struct output_capacity_short {
  bool operator()(const runtime::encode_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !output_capacity_covers_text{}(ev, ctx);
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
    return ctx.rwkv_tables_ready && ctx.rwkv_vocab == ctx.vocab;
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

struct unk_lookup_found {
  bool operator()(const runtime::encode_runtime & ev, const action::context &) const noexcept {
    return ev.unk_lookup_found;
  }
};

struct unk_lookup_missing {
  bool operator()(const runtime::encode_runtime & ev, const action::context &) const noexcept {
    return !ev.unk_lookup_found;
  }
};

struct encode_push_failed {
  bool operator()(const runtime::encode_runtime & ev, const action::context &) const noexcept {
    return ev.encode_push_failed;
  }
};

struct encode_push_ok {
  bool operator()(const runtime::encode_runtime & ev, const action::context &) const noexcept {
    return !ev.encode_push_failed;
  }
};

}  // namespace emel::text::encoders::rwkv::guard
