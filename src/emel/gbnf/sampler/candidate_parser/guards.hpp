#pragma once

#include "emel/gbnf/sampler/candidate_parser/context.hpp"
#include "emel/gbnf/sampler/candidate_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::candidate_parser::guard {

struct has_apply_text {
  bool operator()(const sampler::event::apply_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           !ev.request.text.empty();
  }
};

struct has_empty_apply_text {
  bool operator()(const sampler::event::apply_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.request.text.empty();
  }
};

struct parse_failed {
  bool operator()(const sampler::event::apply_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           !has_apply_text{}(ev, ctx) &&
           !has_empty_apply_text{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::sampler::candidate_parser::guard
