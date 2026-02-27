#pragma once

#include "emel/gbnf/sampler/accept_parser/context.hpp"
#include "emel/gbnf/sampler/accept_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::accept_parser::action {

struct consume_accepted {
  void operator()(const sampler::event::sample_runtime & ev, const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.accept_result = events::accept_result::accepted;
  }
};

struct consume_rejected {
  void operator()(const sampler::event::sample_runtime & ev, const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.accept_result = events::accept_result::rejected;
  }
};

struct dispatch_parse_failed {
  void operator()(const sampler::event::sample_runtime & ev, const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::parse_failed);
    ev.ctx.accept_result = events::accept_result::unknown;
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(sampler::error::internal_error);
      if constexpr (requires { ev.ctx.accept_result; }) {
        ev.ctx.accept_result = events::accept_result::unknown;
      }
    }
  }
};

inline constexpr consume_accepted consume_accepted{};
inline constexpr consume_rejected consume_rejected{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::accept_parser::action
