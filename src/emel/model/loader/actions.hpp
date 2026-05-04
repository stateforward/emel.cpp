#pragma once

#include "emel/io/loader/sm.hpp"
#include "emel/model/loader/context.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::action {

namespace err = emel::error;

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type &ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline void
reset_tensor_bind_events(event::tensor_phase_events &events) noexcept {
  events.bind_done.raised = false;
  events.bind_error.raised = false;
  events.bind_error.err = emel::error::cast(emel::model::tensor::error::none);
}

inline void record_bind_done_event(
    void *object, const emel::model::tensor::events::bind_done &) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->bind_done.raised = true;
  events->bind_error.raised = false;
}

inline void record_bind_error_event(
    void *object, const emel::model::tensor::events::bind_error &ev) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->bind_done.raised = false;
  events->bind_error.raised = true;
  events->bind_error.err = ev.err;
}

inline void
reset_tensor_plan_events(event::tensor_phase_events &events) noexcept {
  events.plan_done.raised = false;
  events.plan_done.effect_count = 0u;
  events.plan_error.raised = false;
  events.plan_error.err = emel::error::cast(emel::model::tensor::error::none);
}

inline void record_plan_done_event(
    void *object, const emel::model::tensor::events::plan_done &ev) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->plan_done.raised = true;
  events->plan_done.effect_count = ev.effect_count;
  events->plan_error.raised = false;
}

inline void record_plan_error_event(
    void *object, const emel::model::tensor::events::plan_error &ev) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->plan_done.raised = false;
  events->plan_done.effect_count = 0u;
  events->plan_error.raised = true;
  events->plan_error.err = ev.err;
}

inline void
reset_tensor_apply_events(event::tensor_phase_events &events) noexcept {
  events.apply_done.raised = false;
  events.apply_error.raised = false;
  events.apply_error.err = emel::error::cast(emel::model::tensor::error::none);
}

inline void record_apply_done_event(
    void *object, const emel::model::tensor::events::apply_done &) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->apply_done.raised = true;
  events->apply_error.raised = false;
}

inline void record_apply_error_event(
    void *object, const emel::model::tensor::events::apply_error &ev) noexcept {
  auto *events = static_cast<event::tensor_phase_events *>(object);
  events->apply_done.raised = false;
  events->apply_error.raised = true;
  events->apply_error.err = ev.err;
}

} // namespace detail

inline void
effect_reset_io_load_events(event::io_phase_events &events,
                            const uint32_t expected_count) noexcept {
  events.load_done.raised = false;
  events.load_done.expected_count = expected_count;
  events.load_done.done_count = 0u;
  events.load_error.raised = false;
  events.load_error.err = emel::error::cast(emel::io::loader::error::none);
}

inline void effect_record_io_load_done_event(
    void *object, const emel::io::loader::events::load_tensor_done &) noexcept {
  auto *events = static_cast<event::io_phase_events *>(object);
  events->load_done.raised = true;
  events->load_done.done_count += 1u;
}

inline void effect_record_io_load_error_event(
    void *object,
    const emel::io::loader::events::load_tensor_error &ev) noexcept {
  auto *events = static_cast<event::io_phase_events *>(object);
  events->load_done.raised = false;
  events->load_error.raised = true;
  events->load_error.err = ev.err;
}

namespace detail {

inline emel::error::type mask_if(const bool predicate) noexcept {
  return emel::error::type{0u} - static_cast<emel::error::type>(predicate);
}

inline emel::error::type
map_tensor_error(const emel::error::type tensor_err) noexcept {
  const auto tensor_none = emel::error::cast(emel::model::tensor::error::none);
  const auto tensor_invalid =
      emel::error::cast(emel::model::tensor::error::invalid_request);
  const auto tensor_model_invalid =
      emel::error::cast(emel::model::tensor::error::model_invalid);
  const auto tensor_internal =
      emel::error::cast(emel::model::tensor::error::internal_error);
  const auto tensor_untracked =
      emel::error::cast(emel::model::tensor::error::untracked);

  const auto loader_none = emel::error::cast(error::none);
  const auto loader_invalid = emel::error::cast(error::invalid_request);
  const auto loader_model_invalid = emel::error::cast(error::model_invalid);
  const auto loader_internal = emel::error::cast(error::internal_error);
  const auto loader_untracked = emel::error::cast(error::untracked);
  const auto loader_backend = emel::error::cast(error::backend_error);

  const auto known_mask = mask_if(tensor_err == tensor_none) |
                          mask_if(tensor_err == tensor_invalid) |
                          mask_if(tensor_err == tensor_model_invalid) |
                          mask_if(tensor_err == tensor_internal) |
                          mask_if(tensor_err == tensor_untracked);

  return (loader_none & mask_if(tensor_err == tensor_none)) |
         (loader_invalid & mask_if(tensor_err == tensor_invalid)) |
         (loader_model_invalid & mask_if(tensor_err == tensor_model_invalid)) |
         (loader_internal & mask_if(tensor_err == tensor_internal)) |
         (loader_untracked & mask_if(tensor_err == tensor_untracked)) |
         (loader_backend & ~known_mask);
}

} // namespace detail

struct begin_load {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.bytes_total = 0;
    ev.ctx.bytes_done = 0;
    ev.ctx.used_mmap = false;
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
  }
};

struct mark_model_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::model_invalid);
  }
};

struct mark_untracked {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    detail::unwrap_runtime_event(ev).ctx.err = err::cast(error::untracked);
  }
};

struct run_parse {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = ev.request.parse_model(ev.request);
  }
};

struct effect_dispatch_tensor_bind_storage {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    detail::reset_tensor_bind_events(ev.tensor_events);

    emel::model::tensor::event::bind_storage bind{
        std::span<emel::model::data::tensor_record>{
            ev.request.model_data.tensors.data(),
            ev.request.model_data.n_tensors},
    };
    bind.on_done = {&ev.tensor_events, detail::record_bind_done_event};
    bind.on_error = {&ev.tensor_events, detail::record_bind_error_event};
    static_cast<void>(ev.request.tensor_loader->process_event(bind));
  }
};

struct effect_dispatch_tensor_plan_load {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    detail::reset_tensor_plan_events(ev.tensor_events);

    emel::model::tensor::event::plan_load plan{ev.request.effect_requests};
    plan.strategy = ev.request.io_strategy;
    plan.on_done = {&ev.tensor_events, detail::record_plan_done_event};
    plan.on_error = {&ev.tensor_events, detail::record_plan_error_event};
    static_cast<void>(ev.request.tensor_loader->process_event(plan));
  }
};

struct effect_mark_io_strategy_unavailable {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::io_strategy_unavailable);
  }
};

struct effect_dispatch_io_loads {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    const uint32_t effect_count = ev.tensor_events.plan_done.effect_count;
    effect_reset_io_load_events(*ev.io_events, effect_count);

    for (uint32_t index = 0u; index < effect_count; ++index) {
      const auto &effect = ev.request.effect_requests[index];
      const emel::io::loader::event::strategy_policy policy{effect.strategy};
      const emel::io::loader::event::tensor_load_span tensor{
          .tensor_id = effect.tensor_id,
          .file_index = effect.file_index,
          .file_offset = effect.offset,
          .byte_size = effect.size,
          .target = effect.target,
      };
      emel::io::loader::event::load_tensor load{tensor, policy};
      load.on_done = {ev.io_events, effect_record_io_load_done_event};
      load.on_error = {ev.io_events, effect_record_io_load_error_event};
      static_cast<void>(ev.request.io_loader->process_event(load));
    }
  }
};

struct effect_dispatch_tensor_apply_results {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    const uint32_t effect_count = ev.tensor_events.plan_done.effect_count;
    detail::reset_tensor_apply_events(ev.tensor_events);

    for (uint32_t index = 0u; index < effect_count; ++index) {
      ev.request.effect_results[index] = emel::model::tensor::effect_result{
          .kind = ev.request.effect_requests[index].kind,
          .handle = ev.request.effect_requests[index].target,
          .err = emel::error::cast(emel::model::tensor::error::none),
      };
    }

    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{
            ev.request.effect_results.data(), effect_count},
        std::span<emel::model::data::tensor_record>{
            ev.request.model_data.tensors.data(),
            ev.request.model_data.n_tensors},
    };
    apply.on_done = {&ev.tensor_events, detail::record_apply_done_event};
    apply.on_error = {&ev.tensor_events, detail::record_apply_error_event};
    static_cast<void>(ev.request.tensor_loader->process_event(apply));
  }
};

struct effect_dispatch_tensor_apply_error_results {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    const uint32_t effect_count = ev.tensor_events.plan_done.effect_count;
    detail::reset_tensor_apply_events(ev.tensor_events);

    for (uint32_t index = 0u; index < effect_count; ++index) {
      ev.request.effect_results[index] = emel::model::tensor::effect_result{
          .kind = ev.request.effect_requests[index].kind,
          .handle = nullptr,
          .err = emel::error::cast(emel::model::tensor::error::backend_error),
      };
    }

    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{
            ev.request.effect_results.data(), effect_count},
        std::span<emel::model::data::tensor_record>{
            ev.request.model_data.tensors.data(),
            ev.request.model_data.n_tensors},
    };
    apply.on_done = {&ev.tensor_events, detail::record_apply_done_event};
    apply.on_error = {&ev.tensor_events, detail::record_apply_error_event};
    static_cast<void>(ev.request.tensor_loader->process_event(apply));
  }
};

struct effect_publish_tensor_load_done_from_file_image {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.bytes_total = ev.request.file_size;
    ev.ctx.bytes_done = ev.request.file_size;
    ev.ctx.err = emel::error::cast(error::none);
    ev.request.model_data.weights_data = ev.request.file_image;
    ev.request.model_data.weights_size = ev.request.file_size;
    ev.request.model_data.weights_split_count = 1u;
    ev.request.model_data.weights_split_offsets[0] = 0u;
    ev.request.model_data.weights_split_sizes[0] = ev.request.file_size;
  }
};

struct effect_publish_tensor_load_done_from_model_data {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.bytes_total = ev.request.model_data.weights_size;
    ev.ctx.bytes_done = ev.request.model_data.weights_size;
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_tensor_bind_error {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = detail::map_tensor_error(ev.tensor_events.bind_error.err);
  }
};

struct effect_mark_tensor_plan_error {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = detail::map_tensor_error(ev.tensor_events.plan_error.err);
  }
};

struct effect_mark_tensor_apply_error {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = detail::map_tensor_error(ev.tensor_events.apply_error.err);
  }
};

struct run_map_layers {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = ev.request.map_layers(ev.request);
  }
};

struct run_validate_structure {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = ev.request.validate_structure(ev.request);
  }
};

struct run_validate_architecture {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = ev.request.validate_architecture_impl(ev.request);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.request.on_done(events::load_done{
        .request = runtime_ev.request,
        .bytes_total = runtime_ev.ctx.bytes_total,
        .bytes_done = runtime_ev.ctx.bytes_done,
        .used_mmap = runtime_ev.ctx.used_mmap,
    });
  }
};

struct publish_done_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    const auto &runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.request.on_error(events::load_error{
        .request = runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct publish_error_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    static_cast<void>(detail::unwrap_runtime_event(ev));
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr begin_load begin_load{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr mark_model_invalid mark_model_invalid{};
inline constexpr mark_untracked mark_untracked{};
inline constexpr run_parse run_parse{};
inline constexpr effect_dispatch_tensor_bind_storage
    effect_dispatch_tensor_bind_storage{};
inline constexpr effect_dispatch_tensor_plan_load
    effect_dispatch_tensor_plan_load{};
inline constexpr effect_mark_io_strategy_unavailable
    effect_mark_io_strategy_unavailable{};
inline constexpr effect_dispatch_io_loads effect_dispatch_io_loads{};
inline constexpr effect_dispatch_tensor_apply_results
    effect_dispatch_tensor_apply_results{};
inline constexpr effect_dispatch_tensor_apply_error_results
    effect_dispatch_tensor_apply_error_results{};
inline constexpr effect_publish_tensor_load_done_from_file_image
    effect_publish_tensor_load_done_from_file_image{};
inline constexpr effect_publish_tensor_load_done_from_model_data
    effect_publish_tensor_load_done_from_model_data{};
inline constexpr effect_mark_tensor_bind_error effect_mark_tensor_bind_error{};
inline constexpr effect_mark_tensor_plan_error effect_mark_tensor_plan_error{};
inline constexpr effect_mark_tensor_apply_error
    effect_mark_tensor_apply_error{};
inline constexpr run_map_layers run_map_layers{};
inline constexpr run_validate_structure run_validate_structure{};
inline constexpr run_validate_architecture run_validate_architecture{};
inline constexpr publish_done publish_done{};
inline constexpr publish_done_noop publish_done_noop{};
inline constexpr publish_error publish_error{};
inline constexpr publish_error_noop publish_error_noop{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::model::loader::action
