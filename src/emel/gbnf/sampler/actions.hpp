#pragma once

#include "emel/gbnf/sampler/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::action {

struct begin_sample {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.read_index = 0;
    ev.ctx.write_index = 0;
    ev.ctx.current_token_id = -1;
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;
    ev.request.error_out = emel::error::cast(error::none);
  }
};

struct mark_invalid_request {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.request.error_out = ev.ctx.err;
  }
};

struct load_candidate_token {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;
    ev.ctx.current_token_id = (&ev.request.candidate_ids)[ev.ctx.read_index];
  }
};

struct compact_candidate {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    (&ev.request.candidate_ids)[ev.ctx.write_index] = (&ev.request.candidate_ids)[ev.ctx.read_index];
    (&ev.request.candidate_scores)[ev.ctx.write_index] =
        (&ev.request.candidate_scores)[ev.ctx.read_index];
    ev.ctx.write_index += 1;
  }
};

struct advance_candidate_cursor {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.read_index += 1;
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;
  }
};

struct mark_parse_failed {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::parse_failed);
    ev.request.error_out = ev.ctx.err;
    ev.request.candidate_count = ev.ctx.write_index;
  }
};

struct publish_done {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.request.candidate_count = ev.ctx.write_index;
    ev.request.error_out = emel::error::cast(error::none);
  }
};

struct publish_error {
  void operator()(const event::sample_runtime & ev, context &) const noexcept {
    ev.request.candidate_count = ev.ctx.write_index;
    ev.request.error_out = ev.ctx.err;
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
inline constexpr load_candidate_token load_candidate_token{};
inline constexpr compact_candidate compact_candidate{};
inline constexpr advance_candidate_cursor advance_candidate_cursor{};
inline constexpr mark_parse_failed mark_parse_failed{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::action
