#pragma once

#include "emel/encoder/context.hpp"
#include "emel/encoder/events.hpp"

namespace emel::encoder::guard {

struct valid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    (void)ctx;
    if (ev.vocab == nullptr) {
      return false;
    }
    if (ev.token_count_out == nullptr || ev.error_out == nullptr) {
      return false;
    }
    if (ev.token_capacity <= 0) {
      return false;
    }
    if (ev.token_ids == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_encode {
  bool operator()(const event::encode & ev, const action::context & ctx) const noexcept {
    return !valid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

}  // namespace emel::encoder::guard
