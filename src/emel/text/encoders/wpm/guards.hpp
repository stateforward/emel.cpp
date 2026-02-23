#pragma once

#include "emel/text/encoders/guards.hpp"
#include "emel/text/encoders/wpm/context.hpp"

namespace emel::text::encoders::wpm::guard {

struct valid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode{}(ev, ctx);
  }
};

struct invalid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::invalid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::phase_ok{}(ctx);
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::phase_failed{}(ctx);
  }
};

}  // namespace emel::text::encoders::wpm::guard
