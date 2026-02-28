#pragma once

#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::bpe::guard {

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

struct text_non_empty_and_preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return text_non_empty{}(ev) && preprocessed{}(ev);
  }
};

struct text_non_empty_and_not_preprocessed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return text_non_empty{}(ev) && not_preprocessed{}(ev);
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

struct valid_encode_and_vocab_changed {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode_and_vocab_changed{}(ev, ctx);
  }
};

struct valid_encode_and_vocab_unchanged {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode_and_vocab_unchanged{}(ev, ctx);
  }
};

}  // namespace emel::text::encoders::bpe::guard
