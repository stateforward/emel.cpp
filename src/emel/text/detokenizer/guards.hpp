#pragma once

#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::guard {

struct valid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.vocab != nullptr;
  }
};

struct invalid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_detokenize {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return ctx.is_bound && ctx.vocab != nullptr && ev.pending_bytes != nullptr &&
           ev.pending_capacity > 0 && ev.pending_length <= ev.pending_capacity &&
           (ev.output != nullptr || ev.output_capacity == 0) &&
           ev.output_length_out != nullptr && ev.pending_length_out != nullptr &&
           ev.error_out != nullptr;
  }
};

struct invalid_detokenize {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return !valid_detokenize{}(ev, ctx);
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

}  // namespace emel::text::detokenizer::guard
