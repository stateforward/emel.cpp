#pragma once

#include "emel/emel.h"
#include "emel/kernel/context.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::guard {

struct valid_dispatch {
  bool operator()(const event::dispatch_scaffold &, const action::context &) const noexcept {
    return true;
  }
};

struct phase_ok {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.err == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.err != EMEL_OK;
  }
};

struct primary_done {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.err == EMEL_OK && ev.ctx.primary_outcome == event::phase_outcome::done;
  }
};

struct primary_unsupported {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.primary_outcome == event::phase_outcome::unsupported;
  }
};

struct primary_failed {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.primary_outcome == event::phase_outcome::failed;
  }
};

struct secondary_done {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.err == EMEL_OK && ev.ctx.secondary_outcome == event::phase_outcome::done;
  }
};

struct secondary_unsupported {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.secondary_outcome == event::phase_outcome::unsupported;
  }
};

struct secondary_failed {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.secondary_outcome == event::phase_outcome::failed;
  }
};

struct tertiary_done {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.err == EMEL_OK && ev.ctx.tertiary_outcome == event::phase_outcome::done;
  }
};

struct tertiary_unsupported {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.tertiary_outcome == event::phase_outcome::unsupported;
  }
};

struct tertiary_failed {
  bool operator()(const event::dispatch_scaffold & ev, const action::context &) const noexcept {
    return ev.ctx.tertiary_outcome == event::phase_outcome::failed;
  }
};

}  // namespace emel::kernel::guard
