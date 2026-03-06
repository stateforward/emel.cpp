#pragma once

#include "emel/model/weight_loader/context.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::action {

struct exec_bind {
  void operator()(const event::bind_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ctx.tensors = ev.request.tensors.data();
    ctx.tensor_count = static_cast<uint32_t>(ev.request.tensors.size());
    ctx.planned_effects = 0u;
  }
};

struct exec_plan {
  void operator()(const event::plan_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ctx.planned_effects = ctx.tensor_count;
    ev.ctx.effect_count = ctx.planned_effects;
    for (uint32_t i = 0u; i < ctx.planned_effects; ++i) {
      const auto & tensor = ctx.tensors[i];
      ev.request.effects[i] = effect_request{
        .kind = effect_kind::k_none,
        .offset = tensor.file_offset,
        .size = tensor.data_size,
        .target = const_cast<void *>(tensor.data),
      };
    }
  }
};

struct scan_apply_effect_errors {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    uint32_t error_flags = 0u;
    for (const auto & result : ev.request.results) {
      error_flags |= static_cast<uint32_t>(
          result.err != emel::error::cast(error::none));
    }
    ev.ctx.has_effect_errors = error_flags != 0u;
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct exec_apply {
  void operator()(const event::apply_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    for (uint32_t i = 0u; i < ctx.planned_effects; ++i) {
      ctx.tensors[i].data = ev.request.results[i].handle;
    }
    ctx.planned_effects = 0u;
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

struct publish_plan_done {
  void operator()(const event::plan_runtime & ev, context &) const noexcept {
    ev.request.on_done(events::plan_done{
      .request = ev.request,
      .effect_count = ev.ctx.effect_count,
    });
  }
};

struct publish_plan_error {
  void operator()(const event::plan_runtime & ev, context &) const noexcept {
    ev.request.on_error(events::plan_error{
      .request = ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct publish_apply_done {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.request.on_done(events::apply_done{
      .request = ev.request,
    });
  }
};

struct publish_apply_error {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.request.on_error(events::apply_error{
      .request = ev.request,
      .err = ev.ctx.err,
    });
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_capacity {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::backend_error);
  }
};

struct mark_apply_invalid_request {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_apply_backend_error {
  void operator()(const event::apply_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::backend_error);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr exec_bind exec_bind{};
inline constexpr exec_plan exec_plan{};
inline constexpr scan_apply_effect_errors scan_apply_effect_errors{};
inline constexpr exec_apply exec_apply{};
inline constexpr publish_bind_done publish_bind_done{};
inline constexpr publish_bind_error publish_bind_error{};
inline constexpr publish_plan_done publish_plan_done{};
inline constexpr publish_plan_error publish_plan_error{};
inline constexpr publish_apply_done publish_apply_done{};
inline constexpr publish_apply_error publish_apply_error{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_capacity mark_capacity{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_apply_invalid_request mark_apply_invalid_request{};
inline constexpr mark_apply_backend_error mark_apply_backend_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::weight_loader::action
