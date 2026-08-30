#pragma once

#include "emel/cact/loader/context.hpp"
#include "emel/cact/loader/detail.hpp"
#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader::action {

struct begin_probe {
  void operator()(const event::probe_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.geometry_out = {};
  }
};

struct begin_bind {
  void operator()(const event::bind_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct begin_parse {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct mark_probe_invalid_request {
  void operator()(const event::probe_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_bind_invalid_request {
  void operator()(const event::bind_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_parse_invalid_request {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_bind_capacity {
  void operator()(const event::bind_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct mark_parse_capacity {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct exec_probe {
  void operator()(const event::probe_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = loader::detail::probe_geometry(ev.request.file_image, ev.ctx.geometry_out);
    ctx.probed = ev.ctx.geometry_out;
    ctx.tensors = {};
  }
};

struct commit_probe_geometry {
  void operator()(const event::probe_runtime & ev, context &) const noexcept {
    ev.request.geometry_out = ev.ctx.geometry_out;
  }
};

struct exec_bind {
  void operator()(const event::bind_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ctx.tensors = ev.request.tensors;
  }
};

struct exec_parse {
  void operator()(const event::parse_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err =
        loader::detail::parse_directory(ev.request.file_image, ctx.probed, ctx.tensors);
  }
};

struct publish_probe_done {
  void operator()(const event::probe_runtime & ev, context &) const noexcept {
    ev.request.on_done(events::probe_done{
      .request = ev.request,
      .geometry_out = ev.ctx.geometry_out,
    });
  }
};

struct publish_probe_error {
  void operator()(const event::probe_runtime & ev, context &) const noexcept {
    ev.request.on_error(events::probe_error{
      .request = ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct publish_bind_done {
  void operator()(const event::bind_runtime & ev, context &) const noexcept {
    ev.request.on_done(events::bind_done{
      .request = ev.request,
    });
  }
};

struct publish_bind_error {
  void operator()(const event::bind_runtime & ev, context &) const noexcept {
    ev.request.on_error(events::bind_error{
      .request = ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct publish_parse_done {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.request.on_done(events::parse_done{
      .request = ev.request,
    });
  }
};

struct publish_parse_error {
  void operator()(const event::parse_runtime & ev, context &) const noexcept {
    ev.request.on_error(events::parse_error{
      .request = ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; }) {
      ev.event_.ctx.err = emel::error::cast(error::internal_error);
    } else if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr begin_probe begin_probe{};
inline constexpr begin_bind begin_bind{};
inline constexpr begin_parse begin_parse{};
inline constexpr mark_probe_invalid_request mark_probe_invalid_request{};
inline constexpr mark_bind_invalid_request mark_bind_invalid_request{};
inline constexpr mark_parse_invalid_request mark_parse_invalid_request{};
inline constexpr mark_bind_capacity mark_bind_capacity{};
inline constexpr mark_parse_capacity mark_parse_capacity{};
inline constexpr exec_probe exec_probe{};
inline constexpr commit_probe_geometry commit_probe_geometry{};
inline constexpr exec_bind exec_bind{};
inline constexpr exec_parse exec_parse{};
inline constexpr publish_probe_done publish_probe_done{};
inline constexpr publish_probe_error publish_probe_error{};
inline constexpr publish_bind_done publish_bind_done{};
inline constexpr publish_bind_error publish_bind_error{};
inline constexpr publish_parse_done publish_parse_done{};
inline constexpr publish_parse_error publish_parse_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::cact::loader::action
