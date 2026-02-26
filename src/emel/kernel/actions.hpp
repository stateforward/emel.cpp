#pragma once

#include <type_traits>

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/context.hpp"
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

inline void reset_dispatch_context(event::scaffold_ctx & ctx) noexcept {
  ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
  ctx.primary_outcome = event::phase_outcome::unknown;
  ctx.secondary_outcome = event::phase_outcome::unknown;
  ctx.tertiary_outcome = event::phase_outcome::unknown;
}

inline void mark_phase_done(event::phase_outcome & outcome, int32_t & err) noexcept {
  outcome = event::phase_outcome::done;
  err = static_cast<int32_t>(emel::error::cast(error::none));
}

inline void mark_phase_unsupported(event::phase_outcome & outcome, int32_t & err) noexcept {
  outcome = event::phase_outcome::unsupported;
  err = static_cast<int32_t>(emel::error::cast(error::none));
}

inline void mark_phase_failed(event::phase_outcome & outcome, int32_t & err,
                              const error phase_error) noexcept {
  outcome = event::phase_outcome::failed;
  err = static_cast<int32_t>(emel::error::cast(phase_error));
}

inline bool invoke_primary(const event::dispatch_scaffold & ev, context & ctx) noexcept {
  return ctx.x86_64_actor.process_event(ev.request);
}

inline bool invoke_secondary(const event::dispatch_scaffold & ev, context & ctx) noexcept {
  return ctx.aarch64_actor.process_event(ev.request);
}

inline bool invoke_tertiary(const event::dispatch_scaffold & ev, context & ctx) noexcept {
  return ctx.wasm_actor.process_event(ev.request);
}

inline bool invoke_primary(const event::dispatch_op & ev, context & ctx) noexcept {
  return ev.dispatch_primary != nullptr && ev.request != nullptr &&
         ev.dispatch_primary(ctx, ev.request);
}

inline bool invoke_secondary(const event::dispatch_op & ev, context & ctx) noexcept {
  return ev.dispatch_secondary != nullptr && ev.request != nullptr &&
         ev.dispatch_secondary(ctx, ev.request);
}

inline bool invoke_tertiary(const event::dispatch_op & ev, context & ctx) noexcept {
  return ev.dispatch_tertiary != nullptr && ev.request != nullptr &&
         ev.dispatch_tertiary(ctx, ev.request);
}

template <class dispatch_event_type>
inline void finalize_phase(const bool accepted, event::phase_outcome & outcome,
                           int32_t & err) noexcept {
  if (accepted) {
    mark_phase_done(outcome, err);
    return;
  }

  if constexpr (std::is_same_v<dispatch_event_type, event::dispatch_op>) {
    mark_phase_unsupported(outcome, err);
  } else {
    mark_phase_failed(outcome, err, error::internal_error);
  }
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
    using resolved_event_type = std::remove_cvref_t<decltype(dispatch_ev)>;
    detail::finalize_phase<resolved_event_type>(detail::invoke_primary(dispatch_ev, ctx),
                                                dispatch_ev.ctx.primary_outcome,
                                                dispatch_ev.ctx.err);
  }
};

struct request_secondary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    using resolved_event_type = std::remove_cvref_t<decltype(dispatch_ev)>;
    detail::finalize_phase<resolved_event_type>(detail::invoke_secondary(dispatch_ev, ctx),
                                                dispatch_ev.ctx.secondary_outcome,
                                                dispatch_ev.ctx.err);
  }
};

struct request_tertiary {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    using resolved_event_type = std::remove_cvref_t<decltype(dispatch_ev)>;
    detail::finalize_phase<resolved_event_type>(detail::invoke_tertiary(dispatch_ev, ctx),
                                                dispatch_ev.ctx.tertiary_outcome,
                                                dispatch_ev.ctx.err);
  }
};

struct mark_unsupported {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    dispatch_ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::unsupported_op));
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
inline constexpr mark_unsupported mark_unsupported{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::action
