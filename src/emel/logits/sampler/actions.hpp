#pragma once

#include "emel/logits/sampler/context.hpp"
#include "emel/logits/sampler/events.hpp"

namespace emel::logits::sampler::action {

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

struct begin_sample {
  void operator()(const event::sample_logits_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.candidate_count = ev.request.vocab_size;
    ev.ctx.sampler_index = 0;
    ev.ctx.sampler_call_error = emel::error::cast(error::none);
    ev.request.selected_token_out = -1;
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

struct prepare_candidates {
  void operator()(const event::sample_logits_runtime & ev, context &) const noexcept {
    const int32_t vocab_size = ev.request.vocab_size;
    const float * logits = &ev.request.logits;
    int32_t * candidate_ids = &ev.request.candidate_ids;
    float * candidate_scores = &ev.request.candidate_scores;

    for (int32_t i = 0; i < vocab_size; ++i) {
      candidate_ids[i] = i;
      candidate_scores[i] = logits[i];
    }
  }
};

struct apply_sampler {
  void operator()(const event::sample_logits_runtime & ev, context & ctx) const noexcept {
    const event::sampler_fn fn = ctx.sampler_fns[ev.ctx.sampler_index];
    ev.ctx.sampler_call_error = fn(
        ev.request.candidate_ids,
        ev.request.candidate_scores,
        ev.ctx.candidate_count,
        ev.request.selected_token_out);
  }
};

struct mark_sampler_error {
  void operator()(const event::sample_logits_runtime & ev, context &) const noexcept {
    ev.ctx.err = ev.ctx.sampler_call_error;
    ev.request.error_out = ev.ctx.err;
  }
};

struct advance_sampler_index {
  void operator()(const event::sample_logits_runtime & ev, context &) const noexcept {
    ev.ctx.sampler_index += 1;
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
    }
  }
};

inline constexpr begin_sample begin_sample{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr prepare_candidates prepare_candidates{};
inline constexpr apply_sampler apply_sampler{};
inline constexpr mark_sampler_error mark_sampler_error{};
inline constexpr advance_sampler_index advance_sampler_index{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::logits::sampler::action
