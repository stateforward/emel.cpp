#pragma once

#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/model/tensor/context.hpp"
#include "emel/model/tensor/detail.hpp"
#include "emel/model/tensor/errors.hpp"

namespace emel::model::tensor::action {

namespace output {

inline void write_error_code(int32_t &target,
                             const emel::error::type err) noexcept {
  target = static_cast<int32_t>(err);
}

} // namespace output

namespace binding {

inline void reset_storage_binding(context &ctx) noexcept {
  ctx.tensors.active_extent = 0u;
  for (size_t tensor_id = 0u; tensor_id < ctx.tensors.lifecycle.size();
       ++tensor_id) {
    ctx.tensors.lifecycle[tensor_id] = event::lifecycle::unbound;
    ctx.tensors.buffer[tensor_id] = nullptr;
    ctx.tensors.buffer_bytes[tensor_id] = 0u;
    ctx.tensors.file_offset[tensor_id] = 0u;
    ctx.tensors.data_size[tensor_id] = 0u;
    ctx.tensors.file_index[tensor_id] = 0u;
    ctx.tensors.tensor_type[tensor_id] = 0;
  }
}

} // namespace binding

struct begin_bind_tensor {
  void operator()(const detail::bind_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
  }
};

struct begin_evict_tensor {
  void operator()(const detail::evict_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
  }
};

struct begin_capture_tensor_state {
  void operator()(const detail::capture_tensor_state_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
  }
};

struct effect_bind_storage {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    binding::reset_storage_binding(ctx);
    ctx.tensors.active_extent = static_cast<uint32_t>(request.tensors.size());
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      const auto &tensor = request.tensors[tensor_id];
      ctx.tensors.lifecycle[tensor_id] = event::lifecycle::unbound;
      ctx.tensors.buffer[tensor_id] = tensor.data;
      ctx.tensors.buffer_bytes[tensor_id] = 0u;
      ctx.tensors.file_offset[tensor_id] = tensor.file_offset;
      ctx.tensors.data_size[tensor_id] = tensor.data_size;
      ctx.tensors.file_index[tensor_id] = tensor.file_index;
      ctx.tensors.tensor_type[tensor_id] = tensor.type;
    }
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
  }
};

struct effect_plan_load {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      request.effects[tensor_id] = event::effect_request{
          .kind = event::effect_kind::k_none,
          .strategy = emel::io::loader::event::strategy_kind::none,
          .tensor_id = static_cast<int32_t>(tensor_id),
          .file_index = ctx.tensors.file_index[tensor_id],
          .offset = ctx.tensors.file_offset[tensor_id],
          .size = ctx.tensors.data_size[tensor_id],
          .target = const_cast<void *>(ctx.tensors.buffer[tensor_id]),
      };
    }
  }
};

struct effect_plan_io_load {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      request.effects[tensor_id] = event::effect_request{
          .kind = event::effect_kind::k_io_load,
          .strategy = request.strategy,
          .tensor_id = static_cast<int32_t>(tensor_id),
          .file_index = ctx.tensors.file_index[tensor_id],
          .offset = ctx.tensors.file_offset[tensor_id],
          .size = ctx.tensors.data_size[tensor_id],
          .target = const_cast<void *>(ctx.tensors.buffer[tensor_id]),
      };
    }
  }
};

struct effect_apply_results {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      ctx.tensors.lifecycle[tensor_id] = event::lifecycle::resident;
      ctx.tensors.buffer[tensor_id] = request.results[tensor_id].handle;
      ctx.tensors.buffer_bytes[tensor_id] = ctx.tensors.data_size[tensor_id];
    }
  }
};

struct effect_apply_results_with_record_output {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    effect_apply_results{}(ev, ctx);
    const auto &request = tensor::detail::request_event(ev);
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      request.tensors[tensor_id].data = request.results[tensor_id].handle;
    }
  }
};

struct exec_bind_tensor {
  void operator()(const detail::bind_tensor_runtime &ev,
                  context &ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[tensor_id] = event::lifecycle::resident;
    ctx.tensors.buffer[tensor_id] = ev.request.buffer;
    ctx.tensors.buffer_bytes[tensor_id] = ev.request.buffer_bytes;
    ctx.tensors.file_offset[tensor_id] = ev.request.tensor_record.file_offset;
    ctx.tensors.data_size[tensor_id] = ev.request.tensor_record.data_size;
    ctx.tensors.file_index[tensor_id] = ev.request.tensor_record.file_index;
    ctx.tensors.tensor_type[tensor_id] = ev.request.tensor_record.type;
    ev.ctx.accepted = true;
  }
};

struct exec_evict_tensor {
  void operator()(const detail::evict_tensor_runtime &ev,
                  context &ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[tensor_id] = event::lifecycle::evicted;
    ctx.tensors.buffer[tensor_id] = nullptr;
    ctx.tensors.buffer_bytes[tensor_id] = 0u;
    ev.ctx.accepted = true;
  }
};

struct exec_capture_tensor_state {
  void operator()(const detail::capture_tensor_state_runtime &ev,
                  context &ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    event::tensor_state &out = *ev.request.state_out;
    out.lifecycle_state = ctx.tensors.lifecycle[tensor_id];
    out.buffer = ctx.tensors.buffer[tensor_id];
    out.buffer_bytes = ctx.tensors.buffer_bytes[tensor_id];
    out.file_offset = ctx.tensors.file_offset[tensor_id];
    out.data_size = ctx.tensors.data_size[tensor_id];
    out.file_index = ctx.tensors.file_index[tensor_id];
    out.tensor_type = ctx.tensors.tensor_type[tensor_id];
    ev.ctx.accepted = true;
  }
};

struct publish_bind_storage_done {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    request.on_done(events::bind_storage_done{
        .request = request,
    });
  }
};

struct publish_bind_storage_error {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
    request.on_error(events::bind_storage_error{
        .request = request,
        .err = emel::error::cast(error::invalid_request),
    });
  }
};

struct record_bind_storage_done {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
  }
};

struct record_bind_storage_invalid_request {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
  }
};

struct record_bind_storage_invalid_request_and_clear_binding {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    binding::reset_storage_binding(ctx);
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
  }
};

struct record_plan_load_done {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
  }
};

struct publish_plan_load_done {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    request.on_done(events::plan_load_done{
        .request = request,
        .effect_count = ctx.tensors.active_extent,
    });
  }
};

struct record_plan_load_invalid_request {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
  }
};

struct publish_plan_load_invalid_request {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
    request.on_error(events::plan_load_error{
        .request = request,
        .err = emel::error::cast(error::invalid_request),
    });
  }
};

struct record_plan_load_capacity_error {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::capacity);
    runtime_ev.ctx.ok = false;
  }
};

struct publish_plan_load_capacity_error {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::capacity);
    runtime_ev.ctx.ok = false;
    request.on_error(events::plan_load_error{
        .request = request,
        .err = emel::error::cast(error::capacity),
    });
  }
};

struct record_apply_effect_results_done {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
  }
};

struct publish_apply_effect_results_done {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    request.on_done(events::apply_effect_results_done{
        .request = request,
    });
  }
};

struct record_apply_effect_results_invalid_request {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
  }
};

struct publish_apply_effect_results_invalid_request {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
    request.on_error(events::apply_effect_results_error{
        .request = request,
        .err = emel::error::cast(error::invalid_request),
    });
  }
};

struct record_apply_effect_results_backend_error {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    runtime_ev.ctx.ok = false;
  }
};

struct publish_apply_effect_results_backend_error {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    const auto &request = tensor::detail::request_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    runtime_ev.ctx.ok = false;
    request.on_error(events::apply_effect_results_error{
        .request = request,
        .err = emel::error::cast(error::backend_error),
    });
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
  }
};

struct mark_capacity {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::capacity);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
  }
};

struct mark_bulk_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
  }
};

struct publish_done_with_error_code {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    output::write_error_code(*runtime_ev.error_code_out,
                             emel::error::cast(error::none));
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.ok = false;
  }
};

struct publish_error_with_error_code {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &ev, context &) const noexcept {
    auto &runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.ok = false;
    output::write_error_code(*runtime_ev.error_code_out, runtime_ev.ctx.err);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      ev.ctx.ok = false;
    }
    if constexpr (requires { ev.status.err; }) {
      ev.status.err = emel::error::cast(error::internal_error);
      ev.status.ok = false;
    }
  }
};

namespace mapped_load_callbacks {

inline void on_io_mmap_request_done(
    void *object, const emel::io::mmap::events::map_tensor_done &ev) noexcept {
  auto *status =
      static_cast<tensor::detail::request_mapped_load_status *>(object);
  status->io_mmap_ok = true;
  status->mapping_handle = ev.handle;
  status->buffer = ev.buffer;
  status->buffer_bytes = ev.buffer_bytes;
}

inline void on_io_mmap_request_error(
    void *object, const emel::io::mmap::events::map_tensor_error &ev) noexcept {
  auto *status =
      static_cast<tensor::detail::request_mapped_load_status *>(object);
  status->io_mmap_ok = false;
  status->io_mmap_err = ev.err;
}

inline void on_io_mmap_release_done(
    void *object,
    const emel::io::mmap::events::release_mapping_done &) noexcept {
  auto *status =
      static_cast<tensor::detail::release_mapped_load_status *>(object);
  status->io_mmap_ok = true;
}

inline void on_io_mmap_release_error(
    void *object,
    const emel::io::mmap::events::release_mapping_error &ev) noexcept {
  auto *status =
      static_cast<tensor::detail::release_mapped_load_status *>(object);
  status->io_mmap_ok = false;
  status->io_mmap_err = ev.err;
}

} // namespace mapped_load_callbacks

struct effect_begin_request_mapped_load {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.accepted = false;
    ev.status.io_mmap_ok = false;
    ev.status.io_mmap_err = emel::error::cast(emel::io::mmap::error::none);
    ev.status.mapping_handle = emel::io::mmap::k_invalid_mapping_handle;
    ev.status.buffer = nullptr;
    ev.status.buffer_bytes = 0u;
  }
};

struct effect_attempt_request_mapped_load_dispatch {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::mmap::event::map_tensor_request inner{
        .tensor_id = ev.request.tensor_id,
        .file_index = 0u,
        .file_offset = ev.request.file_offset,
        .byte_size = ev.request.byte_size,
        .file_path = ev.request.file_path,
    };
    emel::io::mmap::event::map_tensor inner_event{inner};
    inner_event.on_done = {static_cast<void *>(&ev.status),
                           mapped_load_callbacks::on_io_mmap_request_done};
    inner_event.on_error = {static_cast<void *>(&ev.status),
                            mapped_load_callbacks::on_io_mmap_request_error};
    ctx.io_mmap->process_event(inner_event);
  }
};

struct effect_commit_request_mapped_load {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[id] = event::lifecycle::mmap_resident;
    ctx.tensors.buffer[id] = ev.status.buffer;
    ctx.tensors.buffer_bytes[id] = ev.status.buffer_bytes;
    ctx.tensors.file_offset[id] = ev.request.file_offset;
    ctx.tensors.data_size[id] = ev.request.byte_size;
    ev.status.ok = true;
    ev.status.accepted = true;
  }
};

struct effect_mark_request_mapped_load_invalid_request {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_request_mapped_load_unsupported_io_mmap {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::io_mmap_unsupported);
    ev.status.ok = false;
  }
};

struct effect_mark_request_mapped_load_tensor_already_resident {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::tensor_already_resident);
    ev.status.ok = false;
  }
};

struct effect_mark_request_mapped_load_io_mmap_failed {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::io_mmap_failed);
    ev.status.ok = false;
  }
};

struct effect_publish_request_mapped_load_done {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::request_mapped_load_done{
        .request = ev.request,
        .mapping_handle = ev.status.mapping_handle,
        .buffer = ev.status.buffer,
        .buffer_bytes = ev.status.buffer_bytes,
    });
  }
};

struct effect_record_request_mapped_load_done {
  void operator()(const detail::request_mapped_load_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_request_mapped_load_error {
  void operator()(const detail::request_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::request_mapped_load_error{
        .request = ev.request,
        .err = ev.status.err,
        .io_mmap_err = ev.status.io_mmap_err,
    });
  }
};

struct effect_record_request_mapped_load_error {
  void operator()(const detail::request_mapped_load_runtime &,
                  context &) const noexcept {}
};

struct effect_begin_release_mapped_load {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.accepted = false;
    ev.status.io_mmap_ok = false;
    ev.status.io_mmap_err = emel::error::cast(emel::io::mmap::error::none);
    ev.status.target_handle = emel::io::mmap::k_invalid_mapping_handle;
  }
};

struct effect_attempt_release_mapped_load_dispatch {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &ctx) const noexcept {
    ev.status.target_handle = ev.request.mapping_handle;
    emel::io::mmap::event::release_mapping inner_event{ev.request.tensor_id,
                                                       ev.status.target_handle};
    inner_event.on_done = {static_cast<void *>(&ev.status),
                           mapped_load_callbacks::on_io_mmap_release_done};
    inner_event.on_error = {static_cast<void *>(&ev.status),
                            mapped_load_callbacks::on_io_mmap_release_error};
    ctx.io_mmap->process_event(inner_event);
  }
};

struct effect_commit_release_mapped_load {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[id] = event::lifecycle::evicted;
    ctx.tensors.buffer[id] = nullptr;
    ctx.tensors.buffer_bytes[id] = 0u;
    ev.status.ok = true;
    ev.status.accepted = true;
  }
};

struct effect_mark_release_mapped_load_invalid_request {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_release_mapped_load_unsupported_io_mmap {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::io_mmap_unsupported);
    ev.status.ok = false;
  }
};

struct effect_mark_release_mapped_load_handle_absent {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::tensor_unmapped);
    ev.status.ok = false;
  }
};

struct effect_mark_release_mapped_load_io_mmap_failed {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::io_mmap_failed);
    ev.status.ok = false;
  }
};

struct effect_publish_release_mapped_load_done {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::release_mapped_load_done{
        .request = ev.request,
    });
  }
};

struct effect_record_release_mapped_load_done {
  void operator()(const detail::release_mapped_load_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_release_mapped_load_error {
  void operator()(const detail::release_mapped_load_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::release_mapped_load_error{
        .request = ev.request,
        .err = ev.status.err,
        .io_mmap_err = ev.status.io_mmap_err,
    });
  }
};

struct effect_record_release_mapped_load_error {
  void operator()(const detail::release_mapped_load_runtime &,
                  context &) const noexcept {}
};

inline constexpr begin_bind_tensor begin_bind_tensor{};
inline constexpr begin_evict_tensor begin_evict_tensor{};
inline constexpr begin_capture_tensor_state begin_capture_tensor_state{};
inline constexpr effect_bind_storage effect_bind_storage{};
inline constexpr effect_plan_load effect_plan_load{};
inline constexpr effect_plan_io_load effect_plan_io_load{};
inline constexpr effect_apply_results effect_apply_results{};
inline constexpr effect_apply_results_with_record_output
    effect_apply_results_with_record_output{};
inline constexpr exec_bind_tensor exec_bind_tensor{};
inline constexpr exec_evict_tensor exec_evict_tensor{};
inline constexpr exec_capture_tensor_state exec_capture_tensor_state{};
inline constexpr publish_bind_storage_done publish_bind_storage_done{};
inline constexpr publish_bind_storage_error publish_bind_storage_error{};
inline constexpr record_bind_storage_done record_bind_storage_done{};
inline constexpr record_bind_storage_invalid_request
    record_bind_storage_invalid_request{};
inline constexpr record_bind_storage_invalid_request_and_clear_binding
    record_bind_storage_invalid_request_and_clear_binding{};
inline constexpr record_plan_load_done record_plan_load_done{};
inline constexpr publish_plan_load_done publish_plan_load_done{};
inline constexpr record_plan_load_invalid_request
    record_plan_load_invalid_request{};
inline constexpr publish_plan_load_invalid_request
    publish_plan_load_invalid_request{};
inline constexpr record_plan_load_capacity_error
    record_plan_load_capacity_error{};
inline constexpr publish_plan_load_capacity_error
    publish_plan_load_capacity_error{};
inline constexpr record_apply_effect_results_done
    record_apply_effect_results_done{};
inline constexpr publish_apply_effect_results_done
    publish_apply_effect_results_done{};
inline constexpr record_apply_effect_results_invalid_request
    record_apply_effect_results_invalid_request{};
inline constexpr publish_apply_effect_results_invalid_request
    publish_apply_effect_results_invalid_request{};
inline constexpr record_apply_effect_results_backend_error
    record_apply_effect_results_backend_error{};
inline constexpr publish_apply_effect_results_backend_error
    publish_apply_effect_results_backend_error{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_capacity mark_capacity{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr mark_bulk_invalid_request mark_bulk_invalid_request{};
inline constexpr publish_done publish_done{};
inline constexpr publish_done_with_error_code publish_done_with_error_code{};
inline constexpr publish_error publish_error{};
inline constexpr publish_error_with_error_code publish_error_with_error_code{};
inline constexpr on_unexpected on_unexpected{};
inline constexpr effect_begin_request_mapped_load
    effect_begin_request_mapped_load{};
inline constexpr effect_attempt_request_mapped_load_dispatch
    effect_attempt_request_mapped_load_dispatch{};
inline constexpr effect_commit_request_mapped_load
    effect_commit_request_mapped_load{};
inline constexpr effect_mark_request_mapped_load_invalid_request
    effect_mark_request_mapped_load_invalid_request{};
inline constexpr effect_mark_request_mapped_load_unsupported_io_mmap
    effect_mark_request_mapped_load_unsupported_io_mmap{};
inline constexpr effect_mark_request_mapped_load_tensor_already_resident
    effect_mark_request_mapped_load_tensor_already_resident{};
inline constexpr effect_mark_request_mapped_load_io_mmap_failed
    effect_mark_request_mapped_load_io_mmap_failed{};
inline constexpr effect_publish_request_mapped_load_done
    effect_publish_request_mapped_load_done{};
inline constexpr effect_record_request_mapped_load_done
    effect_record_request_mapped_load_done{};
inline constexpr effect_publish_request_mapped_load_error
    effect_publish_request_mapped_load_error{};
inline constexpr effect_record_request_mapped_load_error
    effect_record_request_mapped_load_error{};
inline constexpr effect_begin_release_mapped_load
    effect_begin_release_mapped_load{};
inline constexpr effect_attempt_release_mapped_load_dispatch
    effect_attempt_release_mapped_load_dispatch{};
inline constexpr effect_commit_release_mapped_load
    effect_commit_release_mapped_load{};
inline constexpr effect_mark_release_mapped_load_invalid_request
    effect_mark_release_mapped_load_invalid_request{};
inline constexpr effect_mark_release_mapped_load_unsupported_io_mmap
    effect_mark_release_mapped_load_unsupported_io_mmap{};
inline constexpr effect_mark_release_mapped_load_handle_absent
    effect_mark_release_mapped_load_handle_absent{};
inline constexpr effect_mark_release_mapped_load_io_mmap_failed
    effect_mark_release_mapped_load_io_mmap_failed{};
inline constexpr effect_publish_release_mapped_load_done
    effect_publish_release_mapped_load_done{};
inline constexpr effect_record_release_mapped_load_done
    effect_record_release_mapped_load_done{};
inline constexpr effect_publish_release_mapped_load_error
    effect_publish_release_mapped_load_error{};
inline constexpr effect_record_release_mapped_load_error
    effect_record_release_mapped_load_error{};

} // namespace emel::model::tensor::action
