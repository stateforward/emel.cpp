#pragma once

#include "emel/emel.h"
#include "emel/kernel/context.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::action {

struct begin_dispatch {
  void operator()(const event::dispatch_scaffold & ev, context & ctx) const noexcept {
    ev.ctx.err = EMEL_OK;
    ev.ctx.primary_outcome = event::phase_outcome::unknown;
    ev.ctx.secondary_outcome = event::phase_outcome::unknown;
    ev.ctx.tertiary_outcome = event::phase_outcome::unknown;
    ++ctx.dispatch_generation;
  }
};

struct request_primary {
  void operator()(const event::dispatch_scaffold & ev, context & ctx) const noexcept {
    (void)ctx.x86_64_actor.process_event(ev.request);
    ev.ctx.primary_outcome = event::phase_outcome::done;
    ev.ctx.err = EMEL_OK;
  }
};

struct request_secondary {
  void operator()(const event::dispatch_scaffold & ev, context & ctx) const noexcept {
    (void)ctx.aarch64_actor.process_event(ev.request);
    ev.ctx.secondary_outcome = event::phase_outcome::done;
    ev.ctx.err = EMEL_OK;
  }
};

struct request_tertiary {
  void operator()(const event::dispatch_scaffold & ev, context & ctx) const noexcept {
    (void)ctx.wasm_actor.process_event(ev.request);
    ev.ctx.tertiary_outcome = event::phase_outcome::done;
    ev.ctx.err = EMEL_OK;
  }
};

struct mark_unsupported {
  void operator()(const event::dispatch_scaffold & ev, context &) const noexcept {
    ev.ctx.err = EMEL_ERR_UNSUPPORTED_OP;
  }
};

struct dispatch_done {
  void operator()(const event::dispatch_scaffold &, const context &) const noexcept {}
};

struct dispatch_error {
  void operator()(const event::dispatch_scaffold &, const context &) const noexcept {}
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = EMEL_ERR_BACKEND;
    }
  }
};

inline constexpr begin_dispatch begin_dispatch{};
inline constexpr request_primary request_primary{};
inline constexpr request_secondary request_secondary{};
inline constexpr request_tertiary request_tertiary{};
inline constexpr mark_unsupported mark_unsupported{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::action
