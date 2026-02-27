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

struct has_more_candidates {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.read_index < ev.request.candidate_count;
  }
};

struct no_more_candidates {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.read_index >= ev.request.candidate_count;
  }
};

struct accept_done {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.accept_result != accept_parser::events::accept_result::unknown;
  }
};

struct accept_failed {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct candidate_accepted {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.accept_result == accept_parser::events::accept_result::accepted;
  }
};

struct candidate_rejected {
  bool operator()(const event::sample_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.accept_result == accept_parser::events::accept_result::rejected;
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
