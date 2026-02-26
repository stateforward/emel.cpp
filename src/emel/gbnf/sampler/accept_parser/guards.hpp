#pragma once

#include "emel/gbnf/sampler/accept_parser/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::accept_parser::guard {

struct valid_accept_token {
  bool operator()(const sampler::event::accept_runtime & ev,
                  const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(sampler::error::none) &&
           ev.request.token_id < ev.request.grammar.rule_count;
  }
};

struct parse_failed {
  bool operator()(const sampler::event::accept_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ev.flow.err == emel::error::cast(sampler::error::none) &&
           !valid_accept_token{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::sampler::accept_parser::guard
