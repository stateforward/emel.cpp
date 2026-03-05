#pragma once

#include "emel/text/jinja/parser/context.hpp"
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

inline bool
callbacks_present(const emel::text::jinja::event::parse &ev) noexcept {
  return static_cast<bool>(ev.dispatch_done) &&
         static_cast<bool>(ev.dispatch_error);
}

} // namespace helper

struct valid_parse {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return helper::valid_parse_request(runtime_ev.request) &&
           helper::callbacks_present(runtime_ev.request);
  }
};

struct invalid_parse_with_callbacks {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return !helper::valid_parse_request(runtime_ev.request) &&
           helper::callbacks_present(runtime_ev.request);
  }
};

struct invalid_parse_without_callbacks {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return !helper::callbacks_present(runtime_ev.request);
  }
};

struct parse_error_none {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::none;
  }
};

struct parse_error_invalid_request {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::invalid_request;
  }
};

struct parse_error_parse_failed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::parse_failed;
  }
};

struct parse_error_internal_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::internal_error;
  }
};

struct parse_error_untracked {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::untracked;
  }
};

struct parse_error_unknown {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    const auto &runtime_ev = helper::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err != error::none &&
           runtime_ev.ctx.err != error::invalid_request &&
           runtime_ev.ctx.err != error::parse_failed &&
           runtime_ev.ctx.err != error::internal_error &&
           runtime_ev.ctx.err != error::untracked;
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
