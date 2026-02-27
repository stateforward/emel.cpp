#pragma once

#include "emel/emel.h"
#include "emel/kernel/context.hpp"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::action {

namespace detail {

template <class dispatch_event_type>
constexpr decltype(auto) unwrap_dispatch_event(const dispatch_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline void reset_dispatch_context(event::dispatch_ctx & ctx) noexcept {
  ctx.primary_accepted = false;
  ctx.secondary_accepted = false;
  ctx.tertiary_accepted = false;
  ctx.quaternary_accepted = false;
  ctx.quinary_accepted = false;
  ctx.senary_accepted = false;

  ctx.primary_outcome = event::phase_outcome::unknown;
  ctx.secondary_outcome = event::phase_outcome::unknown;
  ctx.tertiary_outcome = event::phase_outcome::unknown;
  ctx.quaternary_outcome = event::phase_outcome::unknown;
  ctx.quinary_outcome = event::phase_outcome::unknown;
  ctx.senary_outcome = event::phase_outcome::unknown;

  ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
}

template <class dispatch_event_type>
inline bool invoke_primary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.x86_64_actor.process_event(ev.request);
}

template <class dispatch_event_type>
inline bool invoke_secondary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.aarch64_actor.process_event(ev.request);
}

template <class dispatch_event_type>
inline bool invoke_tertiary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.wasm_actor.process_event(ev.request);
}

template <class dispatch_event_type>
inline bool invoke_quaternary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.cuda_actor.process_event(ev.request);
}

template <class dispatch_event_type>
inline bool invoke_quinary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.metal_actor.process_event(ev.request);
}

template <class dispatch_event_type>
inline bool invoke_senary(const dispatch_event_type & ev, context & ctx) noexcept {
  return ctx.vulkan_actor.process_event(ev.request);
}

}  // namespace detail

struct begin_dispatch {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    detail::reset_dispatch_context(dispatch_ev.ctx);
    ++ctx.dispatch_generation;
  }
};

struct request_primary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.primary_accepted = detail::invoke_primary(dispatch_ev, ctx);
  }
};

struct request_secondary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.secondary_accepted = detail::invoke_secondary(dispatch_ev, ctx);
  }
};

struct request_tertiary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.tertiary_accepted = detail::invoke_tertiary(dispatch_ev, ctx);
  }
};

struct request_quaternary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quaternary_accepted = detail::invoke_quaternary(dispatch_ev, ctx);
  }
};

struct request_quinary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quinary_accepted = detail::invoke_quinary(dispatch_ev, ctx);
  }
};

struct request_senary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.senary_accepted = detail::invoke_senary(dispatch_ev, ctx);
  }
};

struct mark_primary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.primary_outcome = event::phase_outcome::done;
  }
};

struct mark_primary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.primary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_secondary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.secondary_outcome = event::phase_outcome::done;
  }
};

struct mark_secondary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.secondary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_tertiary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.tertiary_outcome = event::phase_outcome::done;
  }
};

struct mark_tertiary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.tertiary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_quaternary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quaternary_outcome = event::phase_outcome::done;
  }
};

struct mark_quaternary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quaternary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_quinary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quinary_outcome = event::phase_outcome::done;
  }
};

struct mark_quinary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.quinary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_senary_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.senary_outcome = event::phase_outcome::done;
  }
};

struct mark_senary_rejected {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.senary_outcome = event::phase_outcome::rejected;
  }
};

struct mark_senary_rejected_unsupported {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.senary_outcome = event::phase_outcome::rejected;
    dispatch_ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::unsupported_op));
  }
};

struct mark_senary_rejected_internal_error {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.senary_outcome = event::phase_outcome::rejected;
    dispatch_ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::internal_error));
  }
};

struct mark_unsupported {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::unsupported_op));
  }
};

struct mark_internal_error {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::internal_error));
  }
};

struct dispatch_done {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &, const context &) const noexcept {}
};

struct dispatch_error {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &, const context &) const noexcept {}
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::internal_error));
    }
  }
};

inline constexpr begin_dispatch begin_dispatch{};
inline constexpr request_primary request_primary{};
inline constexpr request_secondary request_secondary{};
inline constexpr request_tertiary request_tertiary{};
inline constexpr request_quaternary request_quaternary{};
inline constexpr request_quinary request_quinary{};
inline constexpr request_senary request_senary{};
inline constexpr mark_primary_done mark_primary_done{};
inline constexpr mark_primary_rejected mark_primary_rejected{};
inline constexpr mark_secondary_done mark_secondary_done{};
inline constexpr mark_secondary_rejected mark_secondary_rejected{};
inline constexpr mark_tertiary_done mark_tertiary_done{};
inline constexpr mark_tertiary_rejected mark_tertiary_rejected{};
inline constexpr mark_quaternary_done mark_quaternary_done{};
inline constexpr mark_quaternary_rejected mark_quaternary_rejected{};
inline constexpr mark_quinary_done mark_quinary_done{};
inline constexpr mark_quinary_rejected mark_quinary_rejected{};
inline constexpr mark_senary_done mark_senary_done{};
inline constexpr mark_senary_rejected mark_senary_rejected{};
inline constexpr mark_senary_rejected_unsupported mark_senary_rejected_unsupported{};
inline constexpr mark_senary_rejected_internal_error mark_senary_rejected_internal_error{};
inline constexpr mark_unsupported mark_unsupported{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::action
