#pragma once

#include "emel/gbnf/sampler/candidate_parser/context.hpp"
#include "emel/gbnf/sampler/candidate_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::candidate_parser::action {

struct consume_text {
  void operator()(const sampler::event::sample_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.candidate_kind = events::candidate_kind::text;
  }
};

struct consume_empty {
  void operator()(const sampler::event::sample_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.candidate_kind = events::candidate_kind::empty;
  }
};

struct dispatch_parse_failed {
  void operator()(const sampler::event::sample_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::parse_failed);
    ev.ctx.candidate_kind = events::candidate_kind::unknown;
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

inline constexpr consume_text consume_text{};
inline constexpr consume_empty consume_empty{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::candidate_parser::action
