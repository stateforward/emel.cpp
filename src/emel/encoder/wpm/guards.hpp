#pragma once

#include "emel/encoder/guards.hpp"
#include "emel/encoder/wpm/context.hpp"

namespace emel::encoder::wpm::guard {

struct valid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    return emel::encoder::guard::valid_encode{}(ev, ctx);
  }
};

struct invalid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    return emel::encoder::guard::invalid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return emel::encoder::guard::phase_ok{}(ctx);
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return emel::encoder::guard::phase_failed{}(ctx);
  }
};

}  // namespace emel::encoder::wpm::guard
