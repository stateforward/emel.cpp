#pragma once

#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::guard {

namespace helper {

constexpr const event::parse_runtime &
unwrap_runtime_event(const event::parse_runtime &ev) noexcept {
  return ev;
}

template <class wrapped_event_type>
  requires requires(const wrapped_event_type &ev) { ev.event_; }
constexpr decltype(auto)
unwrap_runtime_event(const wrapped_event_type &ev) noexcept {
  return ev.event_;
}

inline bool
valid_parse_request(const emel::text::jinja::event::parse &ev) noexcept {
  return ev.template_text.data() != nullptr && !ev.template_text.empty();
}

} // namespace helper

struct valid_parse {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return helper::valid_parse_request(runtime_ev.request);
  }
};

struct invalid_parse {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return !helper::valid_parse_request(runtime_ev.request);
  }
};

struct phase_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::none;
  }
};

struct phase_failed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &ctx) const noexcept {
    return !phase_ok{}(ev, ctx);
  }
};

struct lexer_has_token {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::none && runtime_ev.ctx.lex_has_token;
  }
};

struct lexer_at_eof {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::none && !runtime_ev.ctx.lex_has_token;
  }
};

} // namespace emel::text::jinja::parser::guard
