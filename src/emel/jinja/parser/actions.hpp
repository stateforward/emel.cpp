#pragma once

#include "emel/emel.h"
#include "emel/jinja/parser/context.hpp"
#include "emel/jinja/parser/events.hpp"

namespace emel::jinja::parser::action {

struct reject_invalid_parse {
  void operator()(const emel::jinja::event::parse & ev,
                  context & ctx) const noexcept {
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.program_out != nullptr) {
      ev.program_out->reset();
    }
    if (ev.dispatch_error) {
      ev.dispatch_error(
          emel::jinja::events::parsing_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
    }
  }
};

struct run_parse {
  void operator()(const emel::jinja::event::parse & ev,
                  context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_FORMAT_UNSUPPORTED;
    ctx.last_error = EMEL_ERR_FORMAT_UNSUPPORTED;

    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_FORMAT_UNSUPPORTED;
    }
    if (ev.program_out != nullptr) {
      ev.program_out->reset();
    }
    if (ev.dispatch_error) {
      ev.dispatch_error(
          emel::jinja::events::parsing_error{&ev, ctx.last_error});
    }
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr reject_invalid_parse reject_invalid_parse{};
inline constexpr run_parse run_parse{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::jinja::parser::action
