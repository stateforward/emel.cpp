#pragma once

#include "emel/cact/loader/context.hpp"
#include "emel/cact/loader/detail.hpp"
#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader::guard {

inline bool
has_file_image(const std::span<const uint8_t> &file_image) noexcept {
  return file_image.data() != nullptr && !file_image.empty();
}

template <class runtime_event_type>
inline emel::error::type runtime_error(const runtime_event_type &ev) noexcept {
  return ev.ctx.err;
}

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
         !error_is(runtime_err, error::internal_error) &&
         !error_is(runtime_err, error::untracked);
}

struct guard_probe_valid_request {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return has_file_image(ev.request.file_image);
  }
};

struct guard_probe_invalid_request {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_probe_valid_request{}(ev, ctx);
  }
};

struct guard_bind_valid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.tensors.data() != nullptr && !ev.request.tensors.empty();
  }
};

struct guard_bind_capacity_sufficient {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return ev.request.tensors.size() >= ctx.probed.num_tensors;
  }
};

struct guard_bind_capacity_insufficient {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return guard_bind_valid_request{}(ev, ctx) &&
           !guard_bind_capacity_sufficient{}(ev, ctx);
  }
};

struct guard_bind_invalid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_valid_request{}(ev, ctx);
  }
};

struct guard_parse_has_file_image {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return has_file_image(ev.request.file_image);
  }
};

struct guard_parse_missing_file_image {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_parse_has_file_image{}(ev, ctx);
  }
};

struct guard_parse_has_bound_storage {
  bool operator()(const event::parse_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.tensors.data() != nullptr;
  }
};

struct guard_parse_missing_bound_storage {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_parse_has_bound_storage{}(ev, ctx);
  }
};

struct guard_parse_bound_capacity_sufficient {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &ctx) const noexcept {
    return guard_parse_has_bound_storage{}(ev, ctx) &&
           ctx.tensors.size() >= ctx.probed.num_tensors;
  }
};

struct guard_parse_bound_capacity_insufficient {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &ctx) const noexcept {
    return guard_parse_has_bound_storage{}(ev, ctx) &&
           !guard_parse_bound_capacity_sufficient{}(ev, ctx);
  }
};

struct guard_probe_error_none {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct guard_probe_error_invalid_request {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct guard_probe_error_model_invalid {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct guard_probe_error_capacity {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct guard_probe_error_parse_failed {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct guard_probe_error_internal_error {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct guard_probe_error_untracked {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct guard_probe_error_unknown {
  bool operator()(const event::probe_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct guard_bind_error_none {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct guard_bind_error_invalid_request {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct guard_bind_error_model_invalid {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct guard_bind_error_capacity {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct guard_bind_error_parse_failed {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct guard_bind_error_internal_error {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct guard_bind_error_untracked {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct guard_bind_error_unknown {
  bool operator()(const event::bind_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct guard_parse_error_none {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct guard_parse_error_invalid_request {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct guard_parse_error_model_invalid {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct guard_parse_error_capacity {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct guard_parse_error_parse_failed {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct guard_parse_error_internal_error {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct guard_parse_error_untracked {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct guard_parse_error_unknown {
  bool operator()(const event::parse_runtime &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

} // namespace emel::cact::loader::guard
