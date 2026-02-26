#pragma once

#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/token_parser/context.hpp"
#include "emel/gbnf/sampler/token_parser/events.hpp"

namespace emel::gbnf::sampler::token_parser::action {

struct consume_text_token {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.token_kind = events::token_kind::text_token;
  }
};

struct consume_empty_token {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::none);
    ev.ctx.token_kind = events::token_kind::empty_token;
  }
};

struct dispatch_parse_failed {
  void operator()(const sampler::event::apply_runtime & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(sampler::error::parse_failed);
    ev.ctx.token_kind = events::token_kind::unknown;
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

inline constexpr consume_text_token consume_text_token{};
inline constexpr consume_empty_token consume_empty_token{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::token_parser::action
