#pragma once

#include "emel/io/read/errors.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/model/tensor/context.hpp"
#include "emel/model/tensor/detail.hpp"
#include "emel/model/tensor/errors.hpp"

namespace emel::model::tensor::guard {

namespace detail {

inline bool valid_tensor_id(const int32_t tensor_id) noexcept {
  return tensor_id >= 0 && tensor_id < tensor::detail::max_tensors;
}

template <class runtime_event_type>
bool operation_dispatched(const runtime_event_type &ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.accepted;
}

} // namespace detail

struct storage_bind_valid {
  template <class event_type>
  bool operator()(const event_type &ev) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.tensors.data() != nullptr && !request.tensors.empty() &&
           request.tensors.size() <=
               static_cast<size_t>(tensor::detail::max_tensors);
  }
};

struct storage_bind_invalid {
  template <class event_type>
  bool operator()(const event_type &ev) const noexcept {
    return !storage_bind_valid{}(ev);
  }
};

struct guard_storage_has_mmap_resident {
  bool operator()(const action::context &ctx) const noexcept {
    for (size_t tensor_id = 0u; tensor_id < ctx.tensors.active_extent;
         ++tensor_id) {
      if (ctx.tensors.lifecycle[tensor_id] == event::lifecycle::mmap_resident) {
        return true;
      }
    }
    return false;
  }
};

struct guard_storage_has_no_mmap_resident {
  bool operator()(const action::context &ctx) const noexcept {
    return !guard_storage_has_mmap_resident{}(ctx);
  }
};

struct guard_storage_bind_valid_without_mmap_resident {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bind_valid{}(ev) &&
           guard_storage_has_no_mmap_resident{}(ctx);
  }
};

struct guard_storage_bind_valid_with_mmap_resident {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bind_valid{}(ev) && guard_storage_has_mmap_resident{}(ctx);
  }
};

struct storage_bound {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.tensors.active_extent != 0u;
  }
};

struct plan_load_has_capacity {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.effects.size() >=
           static_cast<size_t>(ctx.tensors.active_extent);
  }
};

struct plan_load_valid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && plan_load_has_capacity{}(ev, ctx);
  }
};

struct plan_load_strategy_none {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.strategy == emel::io::loader::event::strategy_kind::none;
  }
};

struct plan_load_strategy_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !plan_load_strategy_none{}(ev, ctx);
  }
};

struct plan_load_valid_without_io_strategy {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return plan_load_valid{}(ev, ctx) && plan_load_strategy_none{}(ev, ctx);
  }
};

struct plan_load_valid_with_io_strategy {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return plan_load_valid{}(ev, ctx) && plan_load_strategy_present{}(ev, ctx);
  }
};

struct plan_load_invalid_request {
  template <class event_type>
  bool operator()(const event_type &,
                  const action::context &ctx) const noexcept {
    return !storage_bound{}(ctx);
  }
};

struct plan_load_invalid_capacity {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && !plan_load_has_capacity{}(ev, ctx);
  }
};

struct apply_results_count_matches {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.results.size() ==
           static_cast<size_t>(ctx.tensors.active_extent);
  }
};

struct apply_results_record_output_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.tensors.data() != nullptr && !request.tensors.empty();
  }
};

struct apply_results_record_output_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.tensors.empty();
  }
};

struct apply_results_record_output_has_capacity {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return apply_results_record_output_present{}(ev, ctx) &&
           request.tensors.size() >=
               static_cast<size_t>(ctx.tensors.active_extent);
  }
};

struct apply_results_valid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && apply_results_count_matches{}(ev, ctx) &&
           (apply_results_record_output_absent{}(ev, ctx) ||
            apply_results_record_output_has_capacity{}(ev, ctx));
  }
};

struct apply_results_invalid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_results_valid{}(ev, ctx);
  }
};

struct apply_effect_errors_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    uint32_t error_flags = 0u;
    for (const auto &result : request.results) {
      error_flags |=
          static_cast<uint32_t>(result.err != emel::error::cast(error::none));
    }
    return error_flags != 0u;
  }
};

struct apply_effect_errors_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_errors_present{}(ev, ctx);
  }
};

struct apply_results_valid_with_effect_errors {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid{}(ev, ctx) &&
           apply_effect_errors_present{}(ev, ctx);
  }
};

struct apply_results_valid_without_effect_errors {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid{}(ev, ctx) &&
           apply_effect_errors_absent{}(ev, ctx);
  }
};

struct apply_results_valid_without_effect_errors_with_record_output {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid_without_effect_errors{}(ev, ctx) &&
           apply_results_record_output_present{}(ev, ctx);
  }
};

struct apply_results_valid_without_effect_errors_without_record_output {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid_without_effect_errors{}(ev, ctx) &&
           apply_results_record_output_absent{}(ev, ctx);
  }
};

template <class runtime_event_type>
inline emel::error::type runtime_error(const runtime_event_type &ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.err;
}

template <class runtime_event_type>
inline bool error_is(const runtime_event_type &ev,
                     const emel::error::type expected) noexcept {
  return runtime_error(ev) == expected;
}

template <class runtime_event_type>
inline bool error_is_unknown(const runtime_event_type &ev) noexcept {
  return !error_is(ev, emel::error::cast(error::none)) &&
         !error_is(ev, emel::error::cast(error::invalid_request)) &&
         !error_is(ev, emel::error::cast(error::capacity)) &&
         !error_is(ev, emel::error::cast(error::backend_error)) &&
         !error_is(ev, emel::error::cast(error::model_invalid)) &&
         !error_is(ev, emel::error::cast(error::out_of_memory)) &&
         !error_is(ev, emel::error::cast(error::internal_error)) &&
         !error_is(ev, emel::error::cast(error::untracked));
}

struct error_none {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct error_invalid_request {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct error_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::capacity));
  }
};

struct error_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct error_model_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct error_out_of_memory {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::out_of_memory));
  }
};

struct error_internal_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct error_untracked {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct error_unknown {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(ev);
  }
};

struct bind_storage_done_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_done);
  }
};

struct bind_storage_done_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !bind_storage_done_callback_present{}(ev, ctx);
  }
};

struct bind_storage_error_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_error);
  }
};

struct bind_storage_error_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !bind_storage_error_callback_present{}(ev, ctx);
  }
};

struct plan_load_done_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_done);
  }
};

struct plan_load_done_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !plan_load_done_callback_present{}(ev, ctx);
  }
};

struct plan_load_error_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_error);
  }
};

struct plan_load_error_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !plan_load_error_callback_present{}(ev, ctx);
  }
};

struct apply_effect_results_done_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_done);
  }
};

struct apply_effect_results_done_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_results_done_callback_present{}(ev, ctx);
  }
};

struct apply_effect_results_error_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_error);
  }
};

struct apply_effect_results_error_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_results_error_callback_present{}(ev, ctx);
  }
};

struct error_code_output_present {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return tensor::detail::unwrap_runtime_event(ev).error_code_out != nullptr;
  }
};

struct error_code_output_absent {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return !error_code_output_present{}(ev);
  }
};

struct bind_tensor_request_valid {
  bool operator()(const tensor::detail::bind_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (!detail::valid_tensor_id(ev.request.tensor_id)) {
      return false;
    }
    const auto lifecycle =
        ctx.tensors.lifecycle[static_cast<size_t>(ev.request.tensor_id)];
    return lifecycle != event::lifecycle::mmap_resident &&
           ev.request.buffer != nullptr && ev.request.buffer_bytes > 0u &&
           ev.request.tensor_record.data_size > 0u;
  }
};

struct bind_tensor_request_invalid {
  bool operator()(const tensor::detail::bind_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !bind_tensor_request_valid{}(ev, ctx);
  }
};

struct evict_tensor_request_valid {
  // Reject legacy eviction of mmap_resident tensors: their mapping must be
  // released through `release_mapped_load`/`io::mmap::release_mapping` so the
  // OS mapping and slot are properly torn down. Allowing a legacy evict to
  // null the buffer here would leak the mmap slot and OS mapping. See PR #83
  // review thread PRRT_kwDORRHzJs5_hhby.
  bool operator()(const tensor::detail::evict_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (!detail::valid_tensor_id(ev.request.tensor_id)) {
      return false;
    }
    const auto lifecycle =
        ctx.tensors.lifecycle[static_cast<size_t>(ev.request.tensor_id)];
    return lifecycle != event::lifecycle::unbound &&
           lifecycle != event::lifecycle::mmap_resident;
  }
};

struct evict_tensor_request_invalid {
  bool operator()(const tensor::detail::evict_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !evict_tensor_request_valid{}(ev, ctx);
  }
};

struct capture_tensor_state_request_valid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime &ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.state_out != nullptr;
  }
};

struct capture_tensor_state_request_invalid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !capture_tensor_state_request_valid{}(ev, ctx);
  }
};

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return detail::operation_dispatched(ev) &&
           tensor::detail::unwrap_runtime_event(ev).ctx.err ==
               emel::error::cast(error::none);
  }
};

struct operation_not_dispatched {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return !detail::operation_dispatched(ev);
  }
};

struct request_mapped_load_request_valid {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const int32_t id = ev.request.tensor_id;
    return detail::valid_tensor_id(id) &&
           static_cast<uint32_t>(id) < ctx.tensors.active_extent &&
           !ev.request.file_path.empty() &&
           ev.request.file_path.size() <=
               emel::io::mmap::k_max_file_path_bytes &&
           ev.request.file_path.find('\0') == std::string_view::npos &&
           ev.request.byte_size > 0u && static_cast<bool>(ev.request.on_done);
  }
};

struct request_mapped_load_request_invalid {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_mapped_load_request_valid{}(ev, ctx);
  }
};

struct request_mapped_load_io_mmap_present {
  bool operator()(const tensor::detail::request_mapped_load_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.io_mmap != nullptr;
  }
};

struct request_mapped_load_io_mmap_absent {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_mapped_load_io_mmap_present{}(ev, ctx);
  }
};

struct request_mapped_load_tensor_already_resident {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    return ctx.tensors.lifecycle[id] == event::lifecycle::resident ||
           ctx.tensors.lifecycle[id] == event::lifecycle::mmap_resident;
  }
};

struct request_mapped_load_tensor_unbound {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_mapped_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_mapped_load_io_mmap_succeeded {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_mmap_ok;
  }
};

struct request_mapped_load_io_mmap_failed {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_mapped_load_io_mmap_succeeded{}(ev, ctx);
  }
};

struct request_mapped_load_done_callback_present {
  bool operator()(
      const tensor::detail::request_mapped_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct request_mapped_load_error_callback_present {
  bool operator()(
      const tensor::detail::request_mapped_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct request_mapped_load_error_callback_absent {
  bool operator()(
      const tensor::detail::request_mapped_load_runtime &ev) const noexcept {
    return !request_mapped_load_error_callback_present{}(ev);
  }
};

struct request_read_load_request_valid {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const int32_t id = ev.request.tensor_id;
    return detail::valid_tensor_id(id) &&
           static_cast<uint32_t>(id) < ctx.tensors.active_extent &&
           !ev.request.file_path.empty() &&
           ev.request.file_path.size() <=
               emel::io::read::k_max_file_path_bytes &&
           ev.request.file_path.find('\0') == std::string_view::npos &&
           ev.request.byte_size > 0u && static_cast<bool>(ev.request.on_done) &&
           ctx.tensors.data_size[static_cast<size_t>(id)] >=
               ev.request.byte_size &&
           ev.request.target_buffer != nullptr &&
           ev.request.target_buffer_bytes >= ev.request.byte_size;
  }
};

struct request_read_load_request_invalid {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_read_load_request_valid{}(ev, ctx);
  }
};

struct request_read_load_io_read_present {
  bool operator()(const tensor::detail::request_read_load_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.io_read != nullptr;
  }
};

struct request_read_load_io_read_absent {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_read_load_io_read_present{}(ev, ctx);
  }
};

struct request_read_load_tensor_already_resident {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    return ctx.tensors.lifecycle[id] == event::lifecycle::resident ||
           ctx.tensors.lifecycle[id] == event::lifecycle::mmap_resident;
  }
};

struct request_read_load_tensor_unbound {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_read_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_read_load_io_read_succeeded {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_read.accepted && ev.status.io_read.ok;
  }
};

struct request_read_load_io_read_failed {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_read_load_io_read_succeeded{}(ev, ctx);
  }
};

struct request_read_load_io_read_invalid_request {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_read.err ==
           emel::error::cast(emel::io::read::error::invalid_request);
  }
};

struct request_read_load_io_read_unsupported {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_read.err ==
               emel::error::cast(emel::io::read::error::unsupported_platform) ||
           ev.status.io_read.err ==
               emel::error::cast(emel::io::read::error::unsupported_resource);
  }
};

struct request_read_load_io_read_file_open_failed {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_read.err ==
           emel::error::cast(emel::io::read::error::file_open_failed);
  }
};

struct request_read_load_io_read_file_read_or_other_failed {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_read_load_io_read_failed{}(ev, ctx) &&
           !request_read_load_io_read_invalid_request{}(ev, ctx) &&
           !request_read_load_io_read_unsupported{}(ev, ctx) &&
           !request_read_load_io_read_file_open_failed{}(ev, ctx);
  }
};

struct request_read_load_error_callback_present {
  bool operator()(
      const tensor::detail::request_read_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct request_read_load_error_callback_absent {
  bool operator()(
      const tensor::detail::request_read_load_runtime &ev) const noexcept {
    return !request_read_load_error_callback_present{}(ev);
  }
};

struct request_staged_load_request_valid {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const int32_t id = ev.request.tensor_id;
    return detail::valid_tensor_id(id) &&
           static_cast<uint32_t>(id) < ctx.tensors.active_extent &&
           ev.request.byte_size > 0u &&
           ev.request.stage_chunk_bytes > 0u &&
           ev.request.source_buffer != nullptr &&
           ev.request.file_offset <= ev.request.source_buffer_bytes &&
           ev.request.byte_size <=
               (ev.request.source_buffer_bytes - ev.request.file_offset) &&
           static_cast<bool>(ev.request.on_done) &&
           ctx.tensors.data_size[static_cast<size_t>(id)] >=
               ev.request.byte_size &&
           ev.request.target_buffer != nullptr &&
           ev.request.target_buffer_bytes >= ev.request.byte_size;
  }
};

struct request_staged_load_request_invalid {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_staged_load_request_valid{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_present {
  bool operator()(const tensor::detail::request_staged_load_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.io_staged_read != nullptr;
  }
};

struct request_staged_load_io_staged_read_absent {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_staged_load_io_staged_read_present{}(ev, ctx);
  }
};

struct request_staged_load_tensor_already_resident {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    return ctx.tensors.lifecycle[id] == event::lifecycle::resident ||
           ctx.tensors.lifecycle[id] == event::lifecycle::mmap_resident;
  }
};

struct request_staged_load_tensor_unbound {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_staged_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_succeeded {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_staged_read_ok;
  }
};

struct request_staged_load_io_staged_read_failed {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_staged_load_io_staged_read_succeeded{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_invalid_request {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_staged_read_err ==
               emel::error::cast(emel::io::staged_read::error::invalid_callbacks) ||
           ev.status.io_staged_read_err ==
               emel::error::cast(
                   emel::io::staged_read::error::invalid_stage_contract) ||
           ev.status.io_staged_read_err ==
               emel::error::cast(
                   emel::io::staged_read::error::invalid_target_window) ||
           ev.status.io_staged_read_err ==
               emel::error::cast(emel::io::staged_read::error::null_source_span) ||
           ev.status.io_staged_read_err ==
               emel::error::cast(
                   emel::io::staged_read::error::source_span_size_mismatch) ||
           ev.status.io_staged_read_err ==
               emel::error::cast(
                   emel::io::staged_read::error::insufficient_source_span);
  }
};

struct request_staged_load_io_staged_read_unsupported {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_staged_read_err ==
           emel::error::cast(emel::io::staged_read::error::unsupported_platform);
  }
};

struct request_staged_load_io_staged_read_other_failed {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_staged_load_io_staged_read_failed{}(ev, ctx) &&
           !request_staged_load_io_staged_read_invalid_request{}(ev, ctx) &&
           !request_staged_load_io_staged_read_unsupported{}(ev, ctx);
  }
};

struct request_staged_load_error_callback_present {
  bool operator()(
      const tensor::detail::request_staged_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct request_staged_load_error_callback_absent {
  bool operator()(
      const tensor::detail::request_staged_load_runtime &ev) const noexcept {
    return !request_staged_load_error_callback_present{}(ev);
  }
};

struct release_mapped_load_request_valid {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const int32_t id = ev.request.tensor_id;
    return detail::valid_tensor_id(id) &&
           static_cast<uint32_t>(id) < ctx.tensors.active_extent;
  }
};

struct release_mapped_load_request_invalid {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_mapped_load_request_valid{}(ev, ctx);
  }
};

struct release_mapped_load_io_mmap_present {
  bool operator()(const tensor::detail::release_mapped_load_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.io_mmap != nullptr;
  }
};

struct release_mapped_load_io_mmap_absent {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_mapped_load_io_mmap_present{}(ev, ctx);
  }
};

struct release_mapped_load_handle_present {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    const size_t id = static_cast<size_t>(ev.request.tensor_id);
    return ev.request.mapping_handle !=
               emel::io::mmap::k_invalid_mapping_handle &&
           ctx.tensors.lifecycle[id] == event::lifecycle::mmap_resident;
  }
};

struct release_mapped_load_handle_absent {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_mapped_load_handle_present{}(ev, ctx);
  }
};

struct release_mapped_load_io_mmap_succeeded {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.io_mmap_ok;
  }
};

struct release_mapped_load_io_mmap_failed {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_mapped_load_io_mmap_succeeded{}(ev, ctx);
  }
};

struct release_mapped_load_done_callback_present {
  bool operator()(
      const tensor::detail::release_mapped_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct release_mapped_load_done_callback_absent {
  bool operator()(
      const tensor::detail::release_mapped_load_runtime &ev) const noexcept {
    return !release_mapped_load_done_callback_present{}(ev);
  }
};

struct release_mapped_load_error_callback_present {
  bool operator()(
      const tensor::detail::release_mapped_load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct release_mapped_load_error_callback_absent {
  bool operator()(
      const tensor::detail::release_mapped_load_runtime &ev) const noexcept {
    return !release_mapped_load_error_callback_present{}(ev);
  }
};

struct request_mapped_load_io_mmap_present_request_invalid {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_mapped_load_io_mmap_present{}(ev, ctx) &&
           request_mapped_load_request_invalid{}(ev, ctx);
  }
};

struct request_mapped_load_io_mmap_present_request_valid_already_resident {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_mapped_load_io_mmap_present{}(ev, ctx) &&
           request_mapped_load_request_valid{}(ev, ctx) &&
           request_mapped_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_mapped_load_io_mmap_present_request_valid_tensor_unbound {
  bool operator()(const tensor::detail::request_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_mapped_load_io_mmap_present{}(ev, ctx) &&
           request_mapped_load_request_valid{}(ev, ctx) &&
           request_mapped_load_tensor_unbound{}(ev, ctx);
  }
};

struct request_read_load_io_read_present_request_invalid {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_read_load_io_read_present{}(ev, ctx) &&
           request_read_load_request_invalid{}(ev, ctx);
  }
};

struct request_read_load_io_read_present_request_valid_already_resident {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_read_load_io_read_present{}(ev, ctx) &&
           request_read_load_request_valid{}(ev, ctx) &&
           request_read_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_read_load_io_read_present_request_valid_tensor_unbound {
  bool operator()(const tensor::detail::request_read_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_read_load_io_read_present{}(ev, ctx) &&
           request_read_load_request_valid{}(ev, ctx) &&
           request_read_load_tensor_unbound{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_present_request_invalid {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_staged_load_io_staged_read_present{}(ev, ctx) &&
           request_staged_load_request_invalid{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_present_request_valid_already_resident {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_staged_load_io_staged_read_present{}(ev, ctx) &&
           request_staged_load_request_valid{}(ev, ctx) &&
           request_staged_load_tensor_already_resident{}(ev, ctx);
  }
};

struct request_staged_load_io_staged_read_present_request_valid_tensor_unbound {
  bool operator()(const tensor::detail::request_staged_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return request_staged_load_io_staged_read_present{}(ev, ctx) &&
           request_staged_load_request_valid{}(ev, ctx) &&
           request_staged_load_tensor_unbound{}(ev, ctx);
  }
};

struct release_mapped_load_io_mmap_present_request_invalid {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return release_mapped_load_io_mmap_present{}(ev, ctx) &&
           release_mapped_load_request_invalid{}(ev, ctx);
  }
};

struct release_mapped_load_io_mmap_present_request_valid_handle_absent {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return release_mapped_load_io_mmap_present{}(ev, ctx) &&
           release_mapped_load_request_valid{}(ev, ctx) &&
           release_mapped_load_handle_absent{}(ev, ctx);
  }
};

struct release_mapped_load_io_mmap_present_request_valid_handle_present {
  bool operator()(const tensor::detail::release_mapped_load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return release_mapped_load_io_mmap_present{}(ev, ctx) &&
           release_mapped_load_request_valid{}(ev, ctx) &&
           release_mapped_load_handle_present{}(ev, ctx);
  }
};

} // namespace emel::model::tensor::guard
