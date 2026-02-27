#pragma once

#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/matcher_parser/context.hpp"
#include "emel/gbnf/sampler/token_parser/events.hpp"

namespace emel::gbnf::sampler::matcher_parser::guard {

struct token_text {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.ctx.token_kind == sampler::token_parser::events::token_kind::text_token;
  }
};

struct token_empty {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.ctx.token_kind == sampler::token_parser::events::token_kind::empty_token;
  }
};

struct parse_failed {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           !token_text{}(ev, ctx) &&
           !token_empty{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::sampler::matcher_parser::guard
