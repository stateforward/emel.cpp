#pragma once

#include "emel/cact/loader/context.hpp"
#include "emel/cact/loader/detail.hpp"
#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader::action {

struct effect_begin_probe {
  void operator()(const event::probe_runtime &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.geometry_out = {};
    ctx.probed = {};
    ctx.probed_file_image = {};
    ctx.tensors = {};
  }
};

struct effect_begin_bind {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_begin_parse {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_probe_invalid_request {
  void operator()(const event::probe_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_mark_bind_invalid_request {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_mark_parse_invalid_request {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_mark_bind_capacity {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct effect_mark_parse_capacity {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct effect_exec_probe {
  void operator()(const event::probe_runtime &ev, context &) const noexcept {
    ev.ctx.err = loader::detail::probe_geometry(ev.request.file_image,
                                                ev.ctx.geometry_out);
  }
};

struct effect_commit_probe_geometry {
  void operator()(const event::probe_runtime &ev, context &ctx) const noexcept {
    ev.request.geometry_out = ev.ctx.geometry_out;
    ctx.probed = ev.ctx.geometry_out;
    ctx.probed_file_image = ev.request.file_image;
    ctx.tensors = {};
  }
};

struct effect_exec_bind {
  void operator()(const event::bind_runtime &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ctx.tensors = ev.request.tensors;
  }
};

struct effect_exec_parse {
  void operator()(const event::parse_runtime &ev, context &ctx) const noexcept {
    ev.ctx.err = loader::detail::parse_directory(ev.request.file_image,
                                                 ctx.probed, ctx.tensors);
  }
};

struct effect_publish_probe_done {
  void operator()(const event::probe_runtime &ev, context &) const noexcept {
    ev.request.on_done(events::probe_done{
        .request = ev.request,
        .geometry_out = ev.ctx.geometry_out,
    });
  }
};

struct effect_publish_probe_error {
  void operator()(const event::probe_runtime &ev, context &) const noexcept {
    ev.request.on_error(events::probe_error{
        .request = ev.request,
        .err = ev.ctx.err,
    });
  }
};

struct effect_publish_bind_done {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.request.on_done(events::bind_done{
        .request = ev.request,
    });
  }
};

struct effect_publish_bind_error {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.request.on_error(events::bind_error{
        .request = ev.request,
        .err = ev.ctx.err,
    });
  }
};

struct effect_publish_parse_done {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.request.on_done(events::parse_done{
        .request = ev.request,
    });
  }
};

struct effect_publish_parse_error {
  void operator()(const event::parse_runtime &ev, context &) const noexcept {
    ev.request.on_error(events::parse_error{
        .request = ev.request,
        .err = ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; }) {
      ev.event_.ctx.err = emel::error::cast(error::internal_error);
    } else if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr effect_begin_probe effect_begin_probe{};
inline constexpr effect_begin_bind effect_begin_bind{};
inline constexpr effect_begin_parse effect_begin_parse{};
inline constexpr effect_mark_probe_invalid_request
    effect_mark_probe_invalid_request{};
inline constexpr effect_mark_bind_invalid_request
    effect_mark_bind_invalid_request{};
inline constexpr effect_mark_parse_invalid_request
    effect_mark_parse_invalid_request{};
inline constexpr effect_mark_bind_capacity effect_mark_bind_capacity{};
inline constexpr effect_mark_parse_capacity effect_mark_parse_capacity{};
inline constexpr effect_exec_probe effect_exec_probe{};
inline constexpr effect_commit_probe_geometry effect_commit_probe_geometry{};
inline constexpr effect_exec_bind effect_exec_bind{};
inline constexpr effect_exec_parse effect_exec_parse{};
inline constexpr effect_publish_probe_done effect_publish_probe_done{};
inline constexpr effect_publish_probe_error effect_publish_probe_error{};
inline constexpr effect_publish_bind_done effect_publish_bind_done{};
inline constexpr effect_publish_bind_error effect_publish_bind_error{};
inline constexpr effect_publish_parse_done effect_publish_parse_done{};
inline constexpr effect_publish_parse_error effect_publish_parse_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::cact::loader::action
