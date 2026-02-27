#pragma once

#include <algorithm>

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

struct execute_build {
  void operator()(const event::build_runtime & ev, context &) const noexcept {
    const int32_t vocab_size = ev.request.vocab_size;
    const float * logits = &ev.request.logits;
    int32_t * candidate_ids = &ev.request.candidate_ids;
    float * candidate_scores = &ev.request.candidate_scores;

    for (int32_t i = 0; i < vocab_size; ++i) {
      candidate_ids[i] = i;
      candidate_scores[i] = logits[i];
    }

    float max_score = candidate_scores[0];
    for (int32_t i = 1; i < vocab_size; ++i) {
      max_score = std::max(max_score, candidate_scores[i]);
    }

    for (int32_t i = 0; i < vocab_size; ++i) {
      candidate_scores[i] -= max_score;
    }

    ev.request.candidate_count_out = vocab_size;
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.request.error_out = emel::error::cast(error::none);
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
inline constexpr execute_build execute_build{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::logits::validator::action
