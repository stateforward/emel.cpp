#pragma once

#include "emel/model/needle/context.hpp"
#include "emel/model/needle/detail.hpp"
#include "emel/model/needle/events.hpp"

namespace emel::model::needle::guard {

inline bool error_is(const emel::error::type runtime_err,
                     const error expected) noexcept {
  return runtime_err == emel::error::cast(expected);
}

inline bool error_is_unknown(const emel::error::type runtime_err) noexcept {
  return !error_is(runtime_err, error::none) &&
         !error_is(runtime_err, error::invalid_request) &&
         !error_is(runtime_err, error::geometry_invalid) &&
         !error_is(runtime_err, error::tensor_count_mismatch) &&
         !error_is(runtime_err, error::tensor_dtype_mismatch) &&
         !error_is(runtime_err, error::tensor_shape_mismatch) &&
         !error_is(runtime_err, error::head_manifest_invalid) &&
         !error_is(runtime_err, error::internal_error);
}

struct guard_bind_valid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.tensors.data() != nullptr && !ev.request.tensors.empty() &&
           static_cast<bool>(ev.request.on_done);
  }
};

struct guard_bind_invalid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_valid_request{}(ev, ctx);
  }
};

struct guard_bind_done_callback_present {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct guard_bind_done_callback_absent {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_done_callback_present{}(ev, ctx);
  }
};

struct guard_bind_error_callback_present {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct guard_bind_error_callback_absent {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_error_callback_present{}(ev, ctx);
  }
};

struct guard_bind_error_none {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::none);
  }
};

struct guard_bind_error_invalid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::invalid_request);
  }
};

struct guard_bind_error_geometry_invalid {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::geometry_invalid);
  }
};

struct guard_bind_error_tensor_count_mismatch {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::tensor_count_mismatch);
  }
};

struct guard_bind_error_tensor_dtype_mismatch {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::tensor_dtype_mismatch);
  }
};

struct guard_bind_error_tensor_shape_mismatch {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::tensor_shape_mismatch);
  }
};

struct guard_bind_error_head_manifest_invalid {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::head_manifest_invalid);
  }
};

struct guard_bind_error_internal_error {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(ev.ctx.err, error::internal_error);
  }
};

struct guard_bind_error_unknown {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(ev.ctx.err);
  }
};

} // namespace emel::model::needle::guard
