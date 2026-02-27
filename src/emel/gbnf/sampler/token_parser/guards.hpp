#pragma once

#include "emel/gbnf/sampler/candidate_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/token_parser/context.hpp"

namespace emel::gbnf::sampler::token_parser::guard {

struct candidate_text {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.ctx.candidate_kind == sampler::candidate_parser::events::candidate_kind::text;
  }
};

struct candidate_empty {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.ctx.candidate_kind == sampler::candidate_parser::events::candidate_kind::empty;
  }
};

struct parse_failed {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           !candidate_text{}(ev, ctx) &&
           !candidate_empty{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::sampler::token_parser::guard
