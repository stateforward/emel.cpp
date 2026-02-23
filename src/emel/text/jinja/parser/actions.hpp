#pragma once

#include "emel/emel.h"
#include "emel/text/jinja/lexer.hpp"
#include "emel/text/jinja/parser/context.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::action {

struct reject_invalid_parse {
  void operator()(const emel::text::jinja::event::parse & ev,
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
          emel::text::jinja::events::parsing_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
    }
  }
};

struct run_parse {
  void operator()(const emel::text::jinja::event::parse & ev,
                  context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;

    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.program_out != nullptr) {
      ev.program_out->reset();
    }

    emel::text::jinja::lexer lex;
    emel::text::jinja::lexer_result lex_res = lex.tokenize(ev.template_text);
    if (lex_res.error != EMEL_OK) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      ctx.last_error = ctx.phase_error;
      if (ev.error_out != nullptr) {
        *ev.error_out = ctx.last_error;
      }
      if (ev.program_out != nullptr) {
        ev.program_out->reset();
        ev.program_out->last_error = ctx.last_error;
        ev.program_out->last_error_pos = lex_res.error_pos;
      }
      if (ev.dispatch_error) {
        ev.dispatch_error(
            emel::text::jinja::events::parsing_error{&ev, ctx.last_error});
      }
      return;
    }

    if (ev.program_out == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      ctx.last_error = ctx.phase_error;
      if (ev.error_out != nullptr) {
        *ev.error_out = ctx.last_error;
      }
      if (ev.dispatch_error) {
        ev.dispatch_error(
            emel::text::jinja::events::parsing_error{&ev, ctx.last_error});
      }
      return;
    }

    detail::recursive_descent_parser parser{*ev.program_out};
    if (!parser.parse(lex_res)) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      ctx.last_error = ctx.phase_error;
      if (ev.error_out != nullptr) {
        *ev.error_out = ctx.last_error;
      }
      if (ev.program_out != nullptr) {
        ev.program_out->last_error = ctx.last_error;
        ev.program_out->last_error_pos = parser.error_pos();
      }
      if (ev.dispatch_error) {
        ev.dispatch_error(
            emel::text::jinja::events::parsing_error{&ev, ctx.last_error});
      }
      return;
    }

    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.dispatch_done) {
      ev.dispatch_done(emel::text::jinja::events::parsing_done{&ev});
    }
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr reject_invalid_parse reject_invalid_parse{};
inline constexpr run_parse run_parse{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::jinja::parser::action
