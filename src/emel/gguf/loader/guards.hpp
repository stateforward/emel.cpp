#pragma once

#include "emel/gguf/loader/context.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"

namespace emel::gguf::loader::guard {

inline bool has_file_image(const std::span<const uint8_t> & file_image) noexcept {
  return file_image.data() != nullptr && !file_image.empty();
}

template <class runtime_event_type>
inline emel::error::type runtime_error(const runtime_event_type & ev) noexcept {
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

struct probe_valid_request {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return has_file_image(ev.request.file_image);
  }
};

struct probe_invalid_request {
  bool operator()(const event::probe_runtime & ev, const action::context & ctx) const noexcept {
    return !probe_valid_request{}(ev, ctx);
  }
};

struct bind_valid_request {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return ev.request.kv_arena.data() != nullptr &&
           !ev.request.kv_arena.empty() &&
           ev.request.kv_entries.data() != nullptr &&
           ev.request.tensors.data() != nullptr &&
           !ev.request.tensors.empty();
  }
};

struct bind_capacity_sufficient {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return ev.request.tensors.size() >= ctx.probed.tensor_count &&
           ev.request.kv_entries.size() >= ctx.probed.kv_count &&
           ev.request.kv_arena.size() >= detail::required_kv_arena_bytes(ctx.probed);
  }
};

struct bind_capacity_insufficient {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return bind_valid_request{}(ev, ctx) && !bind_capacity_sufficient{}(ev, ctx);
  }
};

struct bind_invalid_request {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return !bind_valid_request{}(ev, ctx);
  }
};

struct parse_has_file_image {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return has_file_image(ev.request.file_image);
  }
};

struct parse_missing_file_image {
  bool operator()(const event::parse_runtime & ev, const action::context & ctx) const noexcept {
    return !parse_has_file_image{}(ev, ctx);
  }
};

struct parse_has_bound_storage {
  bool operator()(const event::parse_runtime &, const action::context & ctx) const noexcept {
    return ctx.tensors.data() != nullptr &&
           ctx.kv_entries.data() != nullptr &&
           ctx.kv_arena.data() != nullptr;
  }
};

struct parse_missing_bound_storage {
  bool operator()(const event::parse_runtime & ev, const action::context & ctx) const noexcept {
    return !parse_has_bound_storage{}(ev, ctx);
  }
};

struct parse_bound_capacity_sufficient {
  bool operator()(const event::parse_runtime & ev, const action::context & ctx) const noexcept {
    return parse_has_bound_storage{}(ev, ctx) &&
           ctx.tensors.size() >= ctx.probed.tensor_count &&
           ctx.kv_entries.size() >= ctx.probed.kv_count &&
           ctx.kv_arena.size() >= detail::required_kv_arena_bytes(ctx.probed);
  }
};

struct parse_bound_capacity_insufficient {
  bool operator()(const event::parse_runtime & ev, const action::context & ctx) const noexcept {
    return parse_has_bound_storage{}(ev, ctx) &&
           !parse_bound_capacity_sufficient{}(ev, ctx);
  }
};

struct probe_error_none {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct probe_error_invalid_request {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct probe_error_model_invalid {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct probe_error_capacity {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct probe_error_parse_failed {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct probe_error_internal_error {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct probe_error_untracked {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct probe_error_unknown {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct bind_error_none {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct bind_error_invalid_request {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct bind_error_model_invalid {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct bind_error_capacity {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct bind_error_parse_failed {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct bind_error_internal_error {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct bind_error_untracked {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct bind_error_unknown {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

struct parse_error_none {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::none);
  }
};

struct parse_error_invalid_request {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::invalid_request);
  }
};

struct parse_error_model_invalid {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::model_invalid);
  }
};

struct parse_error_capacity {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::capacity);
  }
};

struct parse_error_parse_failed {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::parse_failed);
  }
};

struct parse_error_internal_error {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::internal_error);
  }
};

struct parse_error_untracked {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is(runtime_error(ev), error::untracked);
  }
};

struct parse_error_unknown {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(runtime_error(ev));
  }
};

}  // namespace emel::gguf::loader::guard
