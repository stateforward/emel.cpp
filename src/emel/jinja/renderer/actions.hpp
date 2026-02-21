#pragma once

#include "emel/emel.h"
#include "emel/jinja/renderer/context.hpp"
#include "emel/jinja/renderer/detail.hpp"
#include "emel/jinja/renderer/events.hpp"

namespace emel::jinja::renderer::action {

struct reject_invalid_render {
  void operator()(const emel::jinja::event::render & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.error_pos = 0;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.output_length != nullptr) {
      *ev.output_length = 0;
    }
    if (ev.output_truncated != nullptr) {
      *ev.output_truncated = false;
    }
    if (ev.dispatch_error) {
      ev.dispatch_error(emel::jinja::events::rendering_error{&ev, EMEL_ERR_INVALID_ARGUMENT, 0});
    }
  }
};

struct run_render {
  void operator()(const emel::jinja::event::render & ev, context & ctx) const noexcept {
    const bool ok = detail::run_render(ctx, ev);
    if (!ok) {
      if (ev.dispatch_error) {
        ev.dispatch_error(emel::jinja::events::rendering_error{&ev, ctx.phase_error, ctx.error_pos});
      }
      return;
    }
    if (ev.dispatch_done) {
      ev.dispatch_done(emel::jinja::events::rendering_done{
          &ev,
          ev.output_length != nullptr ? *ev.output_length : 0,
          ev.output_truncated != nullptr ? *ev.output_truncated : false});
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

inline constexpr reject_invalid_render reject_invalid_render{};
inline constexpr run_render run_render{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::jinja::renderer::action
