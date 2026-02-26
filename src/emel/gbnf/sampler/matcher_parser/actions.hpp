#pragma once

#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/matcher_parser/context.hpp"
#include "emel/gbnf/sampler/matcher_parser/events.hpp"

namespace emel::gbnf::sampler::matcher_parser::action {

struct consume_match_accepted {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.candidate_allowed = true;
    ev.ctx.match_result = events::match_result::accepted;
  }
};

struct consume_match_rejected {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.candidate_allowed = false;
    ev.ctx.match_result = events::match_result::rejected;
  }
};

struct dispatch_parse_failed {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::parse_failed);
    ev.ctx.match_result = events::match_result::unknown;
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(sampler::error::internal_error);
    }
  }
};

inline constexpr consume_match_accepted consume_match_accepted{};
inline constexpr consume_match_rejected consume_match_rejected{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::matcher_parser::action
