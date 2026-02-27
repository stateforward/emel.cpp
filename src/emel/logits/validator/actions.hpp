#pragma once

#include "emel/logits/validator/context.hpp"
#include "emel/logits/validator/events.hpp"

namespace emel::logits::validator::action {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace detail

struct begin_build {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.candidate_count = 0;
    ev.ctx.build_cursor = 0;
    ev.ctx.max_cursor = 0;
    ev.ctx.normalize_cursor = 0;
    ev.ctx.max_score = 0.0F;
    ev.request.candidate_count_out = 0;
    ev.request.error_out = emel::error::cast(error::none);
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct prepare_candidates_begin {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.build_cursor = 0;
  }
};

struct prepare_candidate_step {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    (&ev.request.candidate_ids)[ev.ctx.build_cursor] = ev.ctx.build_cursor;
    (&ev.request.candidate_scores)[ev.ctx.build_cursor] =
        (&ev.request.logits)[ev.ctx.build_cursor];
  }
};

struct advance_prepare_cursor {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.build_cursor += 1;
  }
};

struct set_candidate_count_from_vocab {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.candidate_count = ev.request.vocab_size;
  }
};

struct begin_max_scan {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.max_cursor = 1;
    ev.ctx.max_score = (&ev.request.candidate_scores)[0];
  }
};

struct update_max_score {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.max_score = (&ev.request.candidate_scores)[ev.ctx.max_cursor];
  }
};

struct advance_max_cursor {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.max_cursor += 1;
  }
};

struct begin_normalize {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.normalize_cursor = 0;
  }
};

struct normalize_score_step {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    (&ev.request.candidate_scores)[ev.ctx.normalize_cursor] -= ev.ctx.max_score;
  }
};

struct advance_normalize_cursor {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    ev.ctx.normalize_cursor += 1;
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.request.error_out = emel::error::cast(error::none);
    runtime_ev.request.candidate_count_out = runtime_ev.ctx.candidate_count;
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.request.candidate_count_out = 0;
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      if constexpr (requires { ev.request.error_out; }) {
        ev.request.error_out = emel::error::cast(error::internal_error);
      }
      if constexpr (requires { ev.request.candidate_count_out; }) {
        ev.request.candidate_count_out = 0;
      }
    }
  }
};

inline constexpr begin_build begin_build{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr prepare_candidates_begin prepare_candidates_begin{};
inline constexpr prepare_candidate_step prepare_candidate_step{};
inline constexpr advance_prepare_cursor advance_prepare_cursor{};
inline constexpr set_candidate_count_from_vocab set_candidate_count_from_vocab{};
inline constexpr begin_max_scan begin_max_scan{};
inline constexpr update_max_score update_max_score{};
inline constexpr advance_max_cursor advance_max_cursor{};
inline constexpr begin_normalize begin_normalize{};
inline constexpr normalize_score_step normalize_score_step{};
inline constexpr advance_normalize_cursor advance_normalize_cursor{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::logits::validator::action
