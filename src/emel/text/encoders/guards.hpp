#pragma once

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/events.hpp"

namespace emel::text::encoders::guard {

struct valid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    (void)ctx;
    if (&ev.request.vocab == &event::default_encode_vocab()) {
      return false;
    }
    if (ev.request.token_ids.empty()) {
      return false;
    }
    return true;
  }
};

struct invalid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return ev.ctx.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct phase_failed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return ev.ctx.err != emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct text_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return ev.request.text.empty();
  }
};

struct text_non_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return !text_empty{}(ev);
  }
};

struct preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return ev.request.preprocessed;
  }
};

struct not_preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return !preprocessed{}(ev);
  }
};

struct vocab_changed {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return ctx.vocab != &ev.request.vocab;
  }
};

struct vocab_unchanged {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !vocab_changed{}(ev, ctx);
  }
};

}  // namespace emel::text::encoders::guard
