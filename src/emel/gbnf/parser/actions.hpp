#pragma once

#include "emel/emel.h"
#include "emel/gbnf/parser/context.hpp"
#include "emel/gbnf/parser/events.hpp"

#include "emel/gbnf/parser/detail.hpp"

namespace emel::gbnf::parser::action {

struct reject_invalid_parse {
  void operator()(const emel::gbnf::event::parse &ev,
                  context &ctx) const noexcept {
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.grammar_out != nullptr) {
      ev.grammar_out->reset();
    }
    if (ev.dispatch_error) {
      ev.dispatch_error(
          emel::gbnf::events::parsing_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
    }
  }
};

struct run_parse {
  void operator()(const emel::gbnf::event::parse &ev,
                  context &ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;

    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.grammar_out != nullptr) {
      ev.grammar_out->reset();
    }

    detail::recursive_descent_parser parser{ctx, ev.grammar_out};
    if (!parser.parse(ev.grammar_text)) {
      if (ctx.phase_error == EMEL_OK) {
        ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      }
    }

    if (ctx.phase_error == EMEL_OK) {
      ctx.last_error = EMEL_OK;
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      if (ev.dispatch_done) {
        ev.dispatch_done(emel::gbnf::events::parsing_done{&ev});
      }
    } else {
      ctx.last_error = ctx.phase_error;
      if (ev.error_out != nullptr) {
        *ev.error_out = ctx.last_error;
      }
      if (ev.dispatch_error) {
        ev.dispatch_error(
            emel::gbnf::events::parsing_error{&ev, ctx.last_error});
      }
    }
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event &, context &ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr reject_invalid_parse reject_invalid_parse{};
inline constexpr run_parse run_parse{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::gbnf::parser::action
