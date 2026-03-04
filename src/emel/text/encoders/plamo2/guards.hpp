#pragma once

#include "emel/text/encoders/plamo2/context.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::plamo2::guard {

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

struct phase_ok {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_ok{}(ev.event_);
  }
};

struct phase_failed {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_failed{}(ev.event_);
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

struct decoded_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_ok{}(ev) && ev.data_len == 0;
  }
};

struct decoded_non_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_ok{}(ev) && ev.data_len > 0;
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
