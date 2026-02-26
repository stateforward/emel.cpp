#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"

namespace emel::generator::guard {

struct valid_generate {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return ev.request.prompt.data() != nullptr &&
           !ev.request.prompt.empty() &&
           ev.request.max_tokens > 0 &&
           ev.request.max_tokens <= action::MAX_GENERATION_STEPS &&
           static_cast<bool>(ev.request.on_done) &&
           static_cast<bool>(ev.request.on_error);
  }
};

struct invalid_generate {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !valid_generate{}(ev, ctx);
  }
};

struct has_error_out {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return ev.request.error_out != nullptr;
  }
};

struct no_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !has_error_out{}(ev, ctx);
  }
};

struct has_error_callback {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct no_error_callback {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return !has_error_callback{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct phase_failed {
  bool operator()(const event::generate_run & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct decode_should_continue {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_ok{}(ev, ctx) && ev.ctx.tokens_generated < ev.ctx.target_tokens;
  }
};

struct decode_complete {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_ok{}(ev, ctx) && ev.ctx.tokens_generated >= ev.ctx.target_tokens;
  }
};

struct phase_ok_with_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_ok{}(ev, ctx) && has_error_out{}(ev, ctx);
  }
};

struct phase_ok_without_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_ok{}(ev, ctx) && no_error_out{}(ev, ctx);
  }
};

struct phase_failed_with_dispatch_and_error_out {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_failed{}(ev, ctx) &&
           has_error_callback{}(ev, ctx) &&
           has_error_out{}(ev, ctx);
  }
};

struct phase_failed_with_dispatch_only {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_failed{}(ev, ctx) &&
           has_error_callback{}(ev, ctx) &&
           no_error_out{}(ev, ctx);
  }
};

struct phase_failed_with_error_out_only {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_failed{}(ev, ctx) &&
           no_error_callback{}(ev, ctx) &&
           has_error_out{}(ev, ctx);
  }
};

struct phase_failed_without_error_channels {
  bool operator()(const event::generate_run & ev, const action::context & ctx) const noexcept {
    return phase_failed{}(ev, ctx) &&
           no_error_callback{}(ev, ctx) &&
           no_error_out{}(ev, ctx);
  }
};

}  // namespace emel::generator::guard
