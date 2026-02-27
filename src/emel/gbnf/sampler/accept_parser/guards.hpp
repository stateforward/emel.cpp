#pragma once

#include "emel/gbnf/sampler/accept_parser/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::accept_parser::guard {

struct token_accepted_by_grammar {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context & ctx) const noexcept {
    const auto & grammar = ctx.grammar.get();
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           ev.ctx.current_token_id >= 0 &&
           static_cast<uint32_t>(ev.ctx.current_token_id) < grammar.rule_count;
  }
};

struct token_rejected_by_grammar {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(sampler::error::none) &&
           !token_accepted_by_grammar{}(ev, ctx);
  }
};

struct parse_failed {
  bool operator()(const sampler::event::sample_runtime & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(sampler::error::none);
  }
};

}  // namespace emel::gbnf::sampler::accept_parser::guard
