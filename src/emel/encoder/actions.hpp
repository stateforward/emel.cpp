#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::action {

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  bool tables_ready = false;
  bool ugm_ready = false;
  int32_t token_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

namespace detail {

inline void dispatch_done(const event::encode & ev, const int32_t token_count) {
  if (ev.dispatch_done == nullptr || ev.owner_sm == nullptr) {
    return;
  }
  ev.dispatch_done(ev.owner_sm, events::encoding_done{&ev, token_count});
}

inline void dispatch_error(const event::encode & ev, const int32_t err) {
  if (ev.dispatch_error == nullptr || ev.owner_sm == nullptr) {
    return;
  }
  ev.dispatch_error(ev.owner_sm, events::encoding_error{&ev, err});
}

inline int32_t normalize_error(const int32_t phase_error, const int32_t last_error) noexcept {
  if (last_error != EMEL_OK) {
    return last_error;
  }
  if (phase_error != EMEL_OK) {
    return phase_error;
  }
  return EMEL_ERR_BACKEND;
}

}  // namespace detail

struct begin_encode {
  void operator()(const event::encode & ev, context & ctx) const noexcept {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct reject_invalid_encode {
  void operator()(const event::encode & ev, context & ctx) const {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    detail::dispatch_error(ev, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct run_encode {
  void operator()(const event::encode &, context & ctx) const noexcept {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct mark_done {
  void operator()(const event::encode & ev, context & ctx) const {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = ctx.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    detail::dispatch_done(ev, ctx.token_count);
  }
};

struct ensure_last_error {
  void operator()(const event::encode & ev, context & ctx) const {
    ctx.last_error = detail::normalize_error(ctx.phase_error, ctx.last_error);
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = ctx.last_error;
    }
    detail::dispatch_error(ev, ctx.last_error);
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event &, context & ctx) const noexcept {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }

  void operator()(const event::encode & ev, context & ctx) const {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    detail::dispatch_error(ev, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr run_encode run_encode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::encoder::action
