#pragma once

#include "emel/gbnf/sampler/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::guard {

struct valid_apply {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done) &&
           static_cast<bool>(ev.request.on_error) &&
           ev.request.start_rule_id < ev.request.grammar.rule_count;
  }
};

struct invalid_apply {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_apply{}(ev, ctx);
  }
};

struct valid_accept {
  bool operator()(const event::accept_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done) &&
           static_cast<bool>(ev.request.on_error) &&
           ev.request.grammar.rule_count != 0u;
  }
};

struct invalid_accept {
  bool operator()(const event::accept_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_accept{}(ev, ctx);
  }
};

struct phase_ok_apply {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none);
  }
};

struct phase_failed_apply {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct candidate_done {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.candidate_kind != candidate_parser::events::candidate_kind::unknown;
  }
};

struct candidate_failed {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct token_done {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.token_kind != token_parser::events::token_kind::unknown;
  }
};

struct token_failed {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct matcher_done {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.match_result != matcher_parser::events::match_result::unknown;
  }
};

struct matcher_failed {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct phase_ok_accept {
  bool operator()(const event::accept_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none);
  }
};

struct phase_failed_accept {
  bool operator()(const event::accept_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

struct accept_done {
  bool operator()(const event::accept_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(error::none) &&
           ev.flow.accept_result != accept_parser::events::accept_result::unknown;
  }
};

struct accept_failed {
  bool operator()(const event::accept_runtime & ev, const action::context &) const noexcept {
    return ev.flow.err != emel::error::cast(error::none);
  }
};

}  // namespace emel::gbnf::sampler::guard
