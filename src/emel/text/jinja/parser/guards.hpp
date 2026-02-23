#pragma once

#include "emel/text/jinja/parser/context.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::guard {

struct valid_parse {
  bool operator()(const event::parse & ev,
                  const action::context &) const noexcept {
    if (ev.template_text.data() == nullptr || ev.template_text.empty()) {
      return false;
    }
    if (ev.program_out == nullptr || ev.error_out == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_parse {
  bool operator()(const event::parse & ev,
                  const action::context & ctx) const noexcept {
    return !valid_parse{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

} // namespace emel::text::jinja::parser::guard
