#pragma once

#include "emel/gguf/loader/context.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"

namespace emel::gguf::loader::guard {

inline bool has_file_image(const std::span<const uint8_t> & file_image) noexcept {
  return file_image.data() != nullptr && !file_image.empty();
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

struct probe_phase_ok {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct probe_phase_failed {
  bool operator()(const event::probe_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct bind_phase_ok {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct bind_phase_failed {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct parse_phase_ok {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct parse_phase_failed {
  bool operator()(const event::parse_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

}  // namespace emel::gguf::loader::guard
