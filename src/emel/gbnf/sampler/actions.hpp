#pragma once

#include <cstdint>

#include "emel/gbnf/sampler/context.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/events.hpp"

namespace emel::gbnf::sampler::action {

struct reject_invalid_apply {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.request.on_error(events::apply_error{
      ev.request.grammar,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct begin_apply {
  void operator()(const event::apply_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.candidate_allowed = false;
    ev.ctx.candidate_kind = candidate_parser::events::candidate_kind::unknown;
    ev.ctx.token_kind = token_parser::events::token_kind::unknown;
    ev.ctx.match_result = matcher_parser::events::match_result::unknown;
    ctx.active_rule_id = ev.request.start_rule_id;
    ctx.frontier_size = 0;
  }
};

struct prepare_candidate_parse {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.candidate_kind = candidate_parser::events::candidate_kind::unknown;
  }
};

struct prepare_token_parse {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.token_kind = token_parser::events::token_kind::unknown;
  }
};

struct prepare_match_parse {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.match_result = matcher_parser::events::match_result::unknown;
  }
};

struct dispatch_apply_done {
  void operator()(const event::apply_runtime & ev, const context &) const noexcept {
    ev.request.on_done(events::apply_done{
      ev.request.grammar,
      ev.ctx.candidate_allowed,
    });
  }
};

struct dispatch_apply_error {
  void operator()(const event::apply_runtime & ev, const context &) const noexcept {
    ev.request.on_error(events::apply_error{
      ev.request.grammar,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_accept {
  void operator()(const event::accept_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.request.on_error(events::accept_error{
      ev.request.grammar,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct begin_accept {
  void operator()(const event::accept_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accepted = false;
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;
  }
};

struct prepare_accept_parse {
  void operator()(const event::accept_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.accept_result = accept_parser::events::accept_result::unknown;
  }
};

struct dispatch_accept_done {
  void operator()(const event::accept_runtime & ev, const context &) const noexcept {
    ev.request.on_done(events::accept_done{
      ev.request.grammar,
      ev.ctx.accepted,
    });
  }
};

struct dispatch_accept_error {
  void operator()(const event::accept_runtime & ev, const context &) const noexcept {
    ev.request.on_error(events::accept_error{
      ev.request.grammar,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_apply reject_invalid_apply{};
inline constexpr begin_apply begin_apply{};
inline constexpr prepare_candidate_parse prepare_candidate_parse{};
inline constexpr prepare_token_parse prepare_token_parse{};
inline constexpr prepare_match_parse prepare_match_parse{};
inline constexpr dispatch_apply_done dispatch_apply_done{};
inline constexpr dispatch_apply_error dispatch_apply_error{};
inline constexpr reject_invalid_accept reject_invalid_accept{};
inline constexpr begin_accept begin_accept{};
inline constexpr prepare_accept_parse prepare_accept_parse{};
inline constexpr dispatch_accept_done dispatch_accept_done{};
inline constexpr dispatch_accept_error dispatch_accept_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::sampler::action
