#pragma once

#include "emel/encoder/guards.hpp"
#include "emel/encoder/plamo2/context.hpp"

namespace emel::encoder::plamo2::guard {

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

struct not_internal_event {
  template <class Event>
  bool operator()(const Event & ev, const action::context & ctx) const noexcept {
    return emel::encoder::guard::not_internal_event{}(ev, ctx);
  }
};

}  // namespace emel::encoder::plamo2::guard
