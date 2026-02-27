#pragma once

#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::guard {

struct valid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    (void)ev;
    return true;
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
           (ev.output != nullptr || ev.output_capacity == 0);
  }
};

struct invalid_detokenize {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return !valid_detokenize{}(ev, ctx);
  }
};

struct bind_phase_ok {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.error_out == error_code(error::none);
  }
};

struct bind_phase_failed {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.error_out != error_code(error::none);
  }
};

struct detokenize_phase_ok {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.error_out == error_code(error::none);
  }
};

struct detokenize_phase_failed {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.error_out != error_code(error::none);
  }
};

struct has_bind_done_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.dispatch_done != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_bind_done_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return !has_bind_done_callback{}(ev);
  }
};

struct has_bind_error_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.dispatch_error != nullptr &&
           ev.owner_sm != nullptr;
  }
};

struct no_bind_error_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return !has_bind_error_callback{}(ev);
  }
};

struct has_detokenize_done_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.dispatch_done != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_detokenize_done_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return !has_detokenize_done_callback{}(ev);
  }
};

struct has_detokenize_error_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.dispatch_error != nullptr &&
           ev.owner_sm != nullptr;
  }
};

struct no_detokenize_error_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return !has_detokenize_error_callback{}(ev);
  }
};

}  // namespace emel::text::detokenizer::guard
