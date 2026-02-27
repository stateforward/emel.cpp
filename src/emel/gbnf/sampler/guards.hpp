#pragma once

#include "emel/gbnf/sampler/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::guard {

struct valid_sample_request {
  bool operator()(const event::sample_runtime & ev, const action::context & ctx) const noexcept {
    const auto & grammar = ctx.grammar.get();
    return ev.request.candidate_count > 0 &&
           grammar.rule_count > 0u &&
           ctx.start_rule_id < grammar.rule_count;
  }
};

struct invalid_sample_request {
  bool operator()(const event::sample_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_sample_request{}(ev, ctx);
  }
};

struct filtered_candidates_available {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.write_index > 0;
  }
};

struct no_filtered_candidates {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.write_index == 0;
  }
};

}  // namespace emel::gbnf::sampler::guard
