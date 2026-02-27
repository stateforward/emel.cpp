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
    ev.ctx.candidate_kind = candidate_parser::events::candidate_kind::unknown;
    ev.ctx.token_kind = token_parser::events::token_kind::unknown;
    ev.ctx.match_result = matcher_parser::events::match_result::unknown;
    ev.ctx.candidate_allowed = false;
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

struct filter_candidates {
  void operator()(const event::sample_runtime & ev, const context & ctx) const noexcept {
    const auto & grammar = ctx.grammar.get();
    const int32_t candidate_count = ev.request.candidate_count;
    int32_t * candidate_ids = &ev.request.candidate_ids;
    float * candidate_scores = &ev.request.candidate_scores;

    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.read_index = candidate_count;
    ev.ctx.write_index = 0;
    ev.ctx.current_token_id = -1;
    ev.ctx.candidate_kind = candidate_parser::events::candidate_kind::unknown;
    ev.ctx.token_kind = token_parser::events::token_kind::unknown;
    ev.ctx.match_result = matcher_parser::events::match_result::unknown;
    ev.ctx.candidate_allowed = false;
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;

    int32_t write_index = 0;
    for (int32_t read_index = 0; read_index < candidate_count; ++read_index) {
      const int32_t token_id = candidate_ids[read_index];
      const bool accepted = token_id >= 0 &&
                            static_cast<uint32_t>(token_id) < grammar.rule_count;
      ev.ctx.current_token_id = token_id;
      ev.ctx.candidate_kind = token_id >= 0 ? candidate_parser::events::candidate_kind::text
                                             : candidate_parser::events::candidate_kind::empty;
      ev.ctx.token_kind = token_id >= 0 ? token_parser::events::token_kind::text_token
                                         : token_parser::events::token_kind::empty_token;
      ev.ctx.accept_result = accepted
                                 ? accept_parser::events::accept_result::accepted
                                 : accept_parser::events::accept_result::rejected;
      ev.ctx.match_result = accepted ? matcher_parser::events::match_result::accepted
                                     : matcher_parser::events::match_result::rejected;
      ev.ctx.candidate_allowed = accepted;
      if (accepted) {
        candidate_ids[write_index] = candidate_ids[read_index];
        candidate_scores[write_index] = candidate_scores[read_index];
        write_index += 1;
      }
    }

    ev.ctx.write_index = write_index;
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
inline constexpr filter_candidates filter_candidates{};
inline constexpr mark_parse_failed mark_parse_failed{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::action
