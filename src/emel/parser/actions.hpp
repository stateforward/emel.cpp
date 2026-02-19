#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/parser/events.hpp"

namespace emel::parser::action {

struct context {
  event::parse_model request = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

inline void store_request(const event::parse_model & ev, context & ctx) noexcept {
  ctx.request = ev;
  ctx.request.owner_sm = nullptr;
  ctx.request.dispatch_done = nullptr;
  ctx.request.dispatch_error = nullptr;
}

inline void clear_request(context & ctx) noexcept {
  ctx.request = {};
}

struct begin_parse {
  void operator()(const event::parse_model & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    store_request(ev, ctx);
  }
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct set_backend_error {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};


struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct clear_request_action {
  void operator()(context & ctx) const noexcept { clear_request(ctx); }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr begin_parse begin_parse{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr clear_request_action clear_request_action{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::parser::action
