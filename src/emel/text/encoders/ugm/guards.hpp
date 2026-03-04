#pragma once

#include "emel/text/encoders/ugm/context.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::ugm::guard {

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

struct normalized_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_ok{}(ev) && ev.normalized.empty();
  }
};

struct normalized_non_empty {
  bool operator()(const runtime::encode_runtime & ev) const noexcept {
    return phase_ok{}(ev) && !ev.normalized.empty();
  }
};

}  // namespace emel::text::encoders::ugm::guard
