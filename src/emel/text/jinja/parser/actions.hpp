#pragma once

#include <cstddef>

#include "emel/callback.hpp"
#include "emel/text/jinja/parser/context.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"

namespace emel::text::jinja::parser::action {

namespace helper {

inline void reset_result(const event::parse &request,
                         event::parse_ctx &ctx) noexcept {
  ctx.err = error::none;
  ctx.error_pos = 0;
  ctx.phase = event::parse_phase::request_validation;
  ctx.statement = event::statement_kind::unknown;
  ctx.expression = event::expression_kind::unknown;
  ctx.token_index = 0;
  ctx.statement_start = 0;
  ctx.expression_start = 0;
  ctx.expression_value_index = 0;
  ctx.lex_cursor = {};
  ctx.lex_token = {};
  ctx.lex_has_token = false;
  ctx.error_out = parser::to_error_code(error::none);
  ctx.error_pos_out = 0;
  ctx.lex_result.tokens.clear();
  ctx.lex_result.error = parser::to_error_code(error::none);
  ctx.lex_result.error_pos = 0;
  ctx.lex_plan_index = 0;

  request.program.body.clear();
  request.program.last_error = parser::to_error_code(error::none);
  request.program.last_error_pos = 0;
}

inline void mark_done(const event::parse &request,
                      event::parse_ctx &ctx) noexcept {
  ctx.err = error::none;
  ctx.error_pos = 0;
  ctx.phase = event::parse_phase::none;
  ctx.error_out = parser::to_error_code(error::none);
  ctx.error_pos_out = 0;
  request.program.last_error = parser::to_error_code(error::none);
  request.program.last_error_pos = 0;
}

inline void mark_error(const event::parse &request, event::parse_ctx &ctx,
                       const error err, const size_t error_pos) noexcept {
  ctx.err = err;
  ctx.error_pos = error_pos;
  ctx.phase = event::parse_phase::none;
  ctx.error_out = parser::to_error_code(err);
  ctx.error_pos_out = error_pos;

  request.program.body.clear();
  request.program.last_error = parser::to_error_code(err);
  request.program.last_error_pos = error_pos;
}

inline void emit_done(const event::parse &request,
                      const event::parse_ctx &) noexcept {
  const events::parsing_done done_ev{
      request,
  };
  (void)request.dispatch_done(done_ev);
}

inline void emit_error(const event::parse &request,
                       const event::parse_ctx &ctx) noexcept {
  const events::parsing_error error_ev{
      request,
      parser::to_error_code(ctx.err),
      ctx.error_pos,
  };
  (void)request.dispatch_error(error_ev);
}

} // namespace helper

inline bool on_lexer_done(
    void *owner,
    const ::emel::text::jinja::lexer::events::next_done &ev) noexcept {
  auto *ctx = static_cast<event::parse_ctx *>(owner);
  ctx->err = error::none;
  ctx->error_pos = 0;
  ctx->lex_result.error = parser::to_error_code(error::none);
  ctx->lex_result.error_pos = 0;
  ctx->lex_token = ev.token;
  ctx->lex_has_token = ev.has_token;
  ctx->lex_cursor = ev.next_cursor;
  return true;
}

inline bool on_lexer_error(
    void *owner,
    const ::emel::text::jinja::lexer::events::next_error &ev) noexcept {
  auto *ctx = static_cast<event::parse_ctx *>(owner);
  ctx->err = static_cast<error>(static_cast<emel::error::type>(ev.err));
  ctx->error_pos = ev.error_pos;
  ctx->lex_result.error = ev.err;
  ctx->lex_result.error_pos = ev.error_pos;
  ctx->lex_token = {};
  ctx->lex_has_token = false;
  return true;
}

namespace runtime_detail {

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

} // namespace runtime_detail

struct reject_invalid_parse {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    helper::mark_error(runtime_ev.request, runtime_ev.ctx,
                       error::invalid_request, 0);
  }
};

struct begin_parse {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    helper::reset_result(runtime_ev.request, runtime_ev.ctx);
  }
};

struct begin_tokenization {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.phase = event::parse_phase::tokenization;
    runtime_ev.ctx.lex_result.tokens.clear();
    runtime_ev.ctx.lex_result.error = parser::to_error_code(error::none);
    runtime_ev.ctx.lex_result.error_pos = 0;
    runtime_ev.ctx.lex_plan_index = 0;
    runtime_ev.ctx.lex_cursor = ::emel::text::jinja::lexer::cursor{
        runtime_ev.ctx.lex_result.source,
        0,
        0,
        0,
        ::emel::text::jinja::token_type::close_statement,
        false,
        false,
    };
    runtime_ev.ctx.lex_token = {};
    runtime_ev.ctx.lex_has_token = false;
  }
};

struct request_next_lex_token {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &ctx) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    const auto &scan = runtime_ev.ctx.lex_plan[runtime_ev.ctx.lex_plan_index];
    runtime_ev.ctx.lex_plan_index += 1;

    runtime_ev.ctx.err = error::internal_error;
    runtime_ev.ctx.error_pos = 0;
    runtime_ev.ctx.lex_token = {};
    runtime_ev.ctx.lex_has_token = false;
    const callback<bool(const ::emel::text::jinja::lexer::events::next_done &)>
        done_cb{
            &runtime_ev.ctx,
            on_lexer_done,
        };
    const callback<bool(const ::emel::text::jinja::lexer::events::next_error &)>
        error_cb{
            &runtime_ev.ctx,
            on_lexer_error,
        };
    const ::emel::text::jinja::lexer::event::next next_ev{
        runtime_ev.ctx.lex_cursor,
        done_cb,
        error_cb,
    };
    const ::emel::text::jinja::parser::lexer::event::next_runtime
        runtime_next_ev{
            next_ev,
            scan,
        };
    (void)ctx.lexer.process_event(runtime_next_ev);
  }
};

struct append_lex_token {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.lex_result.tokens.push_back(runtime_ev.ctx.lex_token);
  }
};

struct commit_lex_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    helper::mark_error(runtime_ev.request, runtime_ev.ctx, runtime_ev.ctx.err,
                       runtime_ev.ctx.error_pos);
  }
};

struct dispatch_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    helper::emit_done(runtime_ev.request, runtime_ev.ctx);
  }
};

struct dispatch_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    helper::emit_error(runtime_ev.request, runtime_ev.ctx);
  }
};

struct on_unexpected {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    helper::mark_error(ev.request, ev.ctx, error::internal_error,
                       ev.ctx.error_pos_out);
  }

  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr reject_invalid_parse reject_invalid_parse{};
inline constexpr begin_parse begin_parse{};
inline constexpr begin_tokenization begin_tokenization{};
inline constexpr request_next_lex_token request_next_lex_token{};
inline constexpr append_lex_token append_lex_token{};
inline constexpr commit_lex_error commit_lex_error{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::jinja::parser::action
