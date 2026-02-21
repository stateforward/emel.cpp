#pragma once

#include "emel/jinja/renderer/context.hpp"
#include "emel/jinja/renderer/events.hpp"

namespace emel::jinja::renderer::guard {

inline constexpr auto valid_render = [](const emel::jinja::event::render & ev) noexcept {
  return ev.program != nullptr && ev.output != nullptr && ev.output_capacity > 0;
};

inline constexpr auto invalid_render = [](const emel::jinja::event::render & ev) noexcept {
  return !valid_render(ev);
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

struct has_stmt_work {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.statements != nullptr && ctx.statement_index < ctx.statements->size();
  }
};

struct no_stmt_work {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.statements == nullptr || ctx.statement_index >= ctx.statements->size();
  }
};

struct needs_expr {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.pending_expr != nullptr;
  }
};

struct has_pending_expr {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.pending_expr != nullptr;
  }
};

struct needs_write {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.pending_value_ready;
  }
};

}  // namespace emel::jinja::renderer::guard
