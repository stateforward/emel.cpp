#pragma once

#include "emel/text/tokenizer/needle/context.hpp"
#include "emel/text/tokenizer/needle/detail.hpp"
#include "emel/text/tokenizer/needle/events.hpp"

namespace emel::text::tokenizer::needle::guard {

inline bool error_is(const emel::error::type runtime_err,
                     const error expected) noexcept {
  return runtime_err == emel::error::cast(expected);
}

inline bool error_is_unknown(const emel::error::type runtime_err) noexcept {
  return !error_is(runtime_err, error::none) &&
         !error_is(runtime_err, error::invalid_request) &&
         !error_is(runtime_err, error::model_invalid) &&
         !error_is(runtime_err, error::capacity) &&
         !error_is(runtime_err, error::parse_failed) &&
         !error_is(runtime_err, error::internal_error);
}

struct guard_load_valid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.blob.data() != nullptr && !ev.request.blob.empty();
  }
};

struct guard_load_invalid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_load_valid_request{}(ev, ctx);
  }
};

struct guard_load_error_none {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::none);
  }
};

struct guard_load_error_invalid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::invalid_request);
  }
};

struct guard_load_error_model_invalid {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::model_invalid);
  }
};

struct guard_load_error_capacity {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::capacity);
  }
};

struct guard_load_error_parse_failed {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::parse_failed);
  }
};

struct guard_load_error_internal_error {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::internal_error);
  }
};

struct guard_load_error_unknown {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(ev.ctx.err);
  }
};

} // namespace emel::text::tokenizer::needle::guard
