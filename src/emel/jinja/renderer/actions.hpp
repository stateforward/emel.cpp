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

struct begin_render {
  void operator()(const emel::jinja::event::render & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.globals = ev.globals;
    ctx.statements = ev.program != nullptr ? &ev.program->body : nullptr;
    ctx.statement_index = 0;
    ctx.pending_expr = nullptr;
    ctx.pending_value = detail::make_undefined();
    ctx.pending_value_ready = false;
    ctx.output = ev.output;
    ctx.output_capacity = ev.output_capacity;
    ctx.output_length = 0;
    ctx.output_truncated = false;

    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.error_pos = 0;
    ctx.steps_remaining = k_max_steps;
    ctx.scope_count = 0;
    ctx.array_items_used = 0;
    ctx.object_entries_used = 0;
    ctx.string_buffer_used = 0;
    ctx.callable_count = 0;
  }
};

struct seed_program {
  void operator()(context & ctx) const noexcept {
    if (ctx.phase_error != EMEL_OK) {
      return;
    }
    if (!detail::push_scope(ctx)) {
      return;
    }
  }
};

struct eval_next_stmt {
  void operator()(context & ctx) const noexcept {
    if (ctx.phase_error != EMEL_OK) {
      return;
    }
    if (ctx.statements == nullptr) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, 0);
      return;
    }
    if (ctx.statement_index >= ctx.statements->size()) {
      return;
    }
    const emel::jinja::ast_node * node = (*ctx.statements)[ctx.statement_index].get();
    ctx.statement_index += 1;
    ctx.pending_expr = nullptr;
    ctx.pending_value_ready = false;

    if (!detail::ensure_steps(ctx, node != nullptr ? node->pos : 0)) {
      return;
    }
    if (node == nullptr) {
      return;
    }
    if (dynamic_cast<const emel::jinja::comment_statement *>(node) != nullptr) {
      return;
    }
    if (dynamic_cast<const emel::jinja::noop_statement *>(node) != nullptr) {
      return;
    }
    if (dynamic_cast<const emel::jinja::break_statement *>(node) != nullptr) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return;
    }
    if (dynamic_cast<const emel::jinja::continue_statement *>(node) != nullptr) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      return;
    }

    detail::render_io io = {};
    detail::init_writer(io, ctx.output, ctx.output_capacity);
    io.writers[0].length = ctx.output_length;
    detail::control_flow flow = detail::control_flow::none;

    if (auto * stmt = dynamic_cast<const emel::jinja::if_statement *>(node)) {
      detail::render_if(ctx, stmt, ctx.globals, io, false, flow);
    } else if (auto * stmt = dynamic_cast<const emel::jinja::for_statement *>(node)) {
      detail::render_for(ctx, stmt, ctx.globals, io, false, flow);
    } else if (auto * stmt = dynamic_cast<const emel::jinja::set_statement *>(node)) {
      detail::render_set(ctx, stmt, ctx.globals, io);
    } else if (auto * stmt = dynamic_cast<const emel::jinja::macro_statement *>(node)) {
      if (ctx.callable_count >= k_max_callables) {
        detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
      } else {
        auto * name_id = dynamic_cast<emel::jinja::identifier *>(stmt->name.get());
        if (name_id == nullptr) {
          detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
        } else {
          callable_slot & slot = ctx.callables[ctx.callable_count++];
          slot.kind = emel::jinja::function_kind::macro;
          slot.macro.stmt = stmt;
          emel::jinja::function_ref ref;
          ref.kind = emel::jinja::function_kind::macro;
          ref.data = &slot;
          detail::set_object_value(ctx, ctx.scopes[ctx.scope_count - 1].locals, name_id->name,
                                   detail::make_function(ref));
        }
      }
    } else if (auto * stmt = dynamic_cast<const emel::jinja::filter_statement *>(node)) {
      detail::render_filter_statement(ctx, stmt, ctx.globals, io);
    } else if (auto * stmt = dynamic_cast<const emel::jinja::call_statement *>(node)) {
      detail::render_call_statement(ctx, stmt, ctx.globals, io);
    } else {
      ctx.pending_expr = node;
      ctx.pending_value = detail::make_undefined();
    }

    ctx.output_length = io.writers[0].length;
    if (flow != detail::control_flow::none) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT, node->pos);
    }
  }
};

struct eval_pending_expr {
  void operator()(context & ctx) const noexcept {
    if (ctx.pending_expr == nullptr || ctx.phase_error != EMEL_OK) {
      return;
    }
    detail::render_io io = {};
    detail::init_writer(io, ctx.output, ctx.output_capacity);
    io.writers[0].length = ctx.output_length;
    ctx.pending_value = detail::eval_expr(ctx, ctx.pending_expr, ctx.globals, io);
    ctx.pending_value_ready = true;
    ctx.pending_expr = nullptr;
    ctx.output_length = io.writers[0].length;
  }
};

struct write_pending_value {
  void operator()(context & ctx) const noexcept {
    if (!ctx.pending_value_ready || ctx.phase_error != EMEL_OK) {
      return;
    }
    detail::render_io io = {};
    detail::init_writer(io, ctx.output, ctx.output_capacity);
    io.writers[0].length = ctx.output_length;
    detail::write_value(ctx, io, ctx.pending_value);
    ctx.output_length = io.writers[0].length;
    ctx.pending_value_ready = false;
    ctx.pending_value = detail::make_undefined();
  }
};

struct finalize_done {
  void operator()(context & ctx) const noexcept {
    const auto * ev = ctx.request;
    if (ev == nullptr) {
      return;
    }
    if (ev->output_length != nullptr) {
      *ev->output_length = ctx.output_length;
    }
    if (ev->output_truncated != nullptr) {
      *ev->output_truncated = ctx.phase_error != EMEL_OK;
    }
    if (ev->error_out != nullptr) {
      *ev->error_out = ctx.phase_error;
    }
    if (ev->error_pos_out != nullptr) {
      *ev->error_pos_out = ctx.error_pos;
    }
    if (ev->dispatch_done) {
      ev->dispatch_done(emel::jinja::events::rendering_done{
          ev,
          ctx.output_length,
          ctx.phase_error != EMEL_OK});
    }
  }
};

struct finalize_error {
  void operator()(context & ctx) const noexcept {
    const auto * ev = ctx.request;
    if (ev == nullptr) {
      return;
    }
    if (ev->output_length != nullptr) {
      *ev->output_length = ctx.output_length;
    }
    if (ev->output_truncated != nullptr) {
      *ev->output_truncated = ctx.phase_error != EMEL_OK;
    }
    if (ev->error_out != nullptr) {
      *ev->error_out = ctx.phase_error;
    }
    if (ev->error_pos_out != nullptr) {
      *ev->error_pos_out = ctx.error_pos;
    }
    if (ev->dispatch_error) {
      ev->dispatch_error(emel::jinja::events::rendering_error{ev, ctx.phase_error, ctx.error_pos});
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

inline constexpr reject_invalid_render reject_invalid_render{};
inline constexpr begin_render begin_render{};
inline constexpr seed_program seed_program{};
inline constexpr eval_next_stmt eval_next_stmt{};
inline constexpr eval_pending_expr eval_pending_expr{};
inline constexpr write_pending_value write_pending_value{};
inline constexpr finalize_done finalize_done{};
inline constexpr finalize_error finalize_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::jinja::renderer::action
