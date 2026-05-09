#pragma once

#include "emel/model/loader/context.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::guard {

struct has_model_path {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !ev.request.model_path.empty();
  }
};

struct has_file_image {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.file_image != nullptr && ev.request.file_size > 0;
  }
};

struct guard_parse_model_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.parse_model);
  }
};

struct valid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return guard_parse_model_present{}(ev) &&
           (has_model_path{}(ev) || has_file_image{}(ev));
  }
};

struct invalid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

inline bool error_is(const event::load_runtime &ev,
                     const emel::error::type expected) noexcept {
  return ev.ctx.err == expected;
}

struct error_none {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct error_invalid_request {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct error_parse_failed {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::parse_failed));
  }
};

struct error_backend_error {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct error_model_invalid {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct error_internal_error {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct error_untracked {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct error_io_strategy_unavailable {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return error_is(ev, emel::error::cast(error::io_strategy_unavailable));
  }
};

struct error_unclassified_code {
  bool operator()(const event::load_runtime &ev) const noexcept {
    const emel::error::type err = ev.ctx.err;
    return err != emel::error::cast(error::none) &&
           err != emel::error::cast(error::invalid_request) &&
           err != emel::error::cast(error::parse_failed) &&
           err != emel::error::cast(error::backend_error) &&
           err != emel::error::cast(error::model_invalid) &&
           err != emel::error::cast(error::internal_error) &&
           err != emel::error::cast(error::untracked) &&
           err != emel::error::cast(error::io_strategy_unavailable);
  }
};

struct should_load_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !ev.request.vocab_only;
  }
};

struct skip_load_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.vocab_only;
  }
};

struct model_has_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.model_data.n_tensors > 0u;
  }
};

struct model_has_no_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !model_has_tensors{}(ev);
  }
};

struct model_tensor_count_within_capacity {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.model_data.n_tensors <=
           static_cast<uint32_t>(emel::model::data::k_max_tensors);
  }
};

struct can_load_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return model_has_tensors{}(ev) &&
           model_tensor_count_within_capacity{}(ev) &&
           ev.request.tensor_loader != nullptr &&
           ev.request.effect_requests.size() >=
               ev.request.model_data.n_tensors &&
           ev.request.effect_results.size() >= ev.request.model_data.n_tensors;
  }
};

struct cannot_load_tensors {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return model_has_tensors{}(ev) && !can_load_tensors{}(ev);
  }
};

struct tensor_bind_done_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.bind_done.raised;
  }
};

struct tensor_bind_error_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.bind_error.raised;
  }
};

struct tensor_bind_unhandled {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !tensor_bind_done_raised{}(ev) && !tensor_bind_error_raised{}(ev);
  }
};

struct tensor_plan_done_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.plan_done.raised;
  }
};

struct io_strategy_none {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_strategy ==
           emel::io::loader::event::strategy_kind::none;
  }
};

struct io_strategy_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !io_strategy_none{}(ev);
  }
};

struct io_strategy_read_copy {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_strategy ==
           emel::io::loader::event::strategy_kind::read_copy;
  }
};

struct io_strategy_staged_read {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_strategy ==
           emel::io::loader::event::strategy_kind::staged_read;
  }
};

struct io_strategy_requires_staging_storage {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_strategy_read_copy{}(ev) || io_strategy_staged_read{}(ev);
  }
};

struct io_strategy_not_read_copy {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !io_strategy_requires_staging_storage{}(ev);
  }
};

struct io_loader_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_loader != nullptr;
  }
};

struct io_loader_absent {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !io_loader_present{}(ev);
  }
};

struct tensor_plan_done_without_io_strategy {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_raised{}(ev) && io_strategy_none{}(ev);
  }
};

struct tensor_plan_done_with_io_strategy_without_loader {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_raised{}(ev) && io_strategy_present{}(ev) &&
           io_loader_absent{}(ev);
  }
};

struct tensor_plan_done_with_io_strategy_with_loader {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_raised{}(ev) && io_strategy_present{}(ev) &&
           io_loader_present{}(ev);
  }
};

struct io_load_batch_span_ready {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_load_spans.size() >=
           ev.tensor_events.plan_done.effect_count;
  }
};

struct io_load_batch_span_missing {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_load_spans.size() <
           ev.tensor_events.plan_done.effect_count;
  }
};

inline uint64_t
read_copy_storage_required_bytes(const event::load_runtime &ev) noexcept {
  uint64_t required = 0u;
  uint32_t index = 0u;
  while (index < ev.tensor_events.plan_done.effect_count) {
    const uint64_t size = ev.request.effect_requests[index].size;
    if (required > static_cast<uint64_t>(-1) - size) {
      return static_cast<uint64_t>(-1);
    }
    required += size;
    ++index;
  }
  return required;
}

struct read_copy_storage_ready {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.read_copy_storage.size() >=
           read_copy_storage_required_bytes(ev);
  }
};

struct read_copy_storage_missing {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.read_copy_storage.size() <
           read_copy_storage_required_bytes(ev);
  }
};

struct tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_with_io_strategy_with_loader{}(ev) &&
           io_strategy_not_read_copy{}(ev) && io_load_batch_span_ready{}(ev);
  }
};

struct tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_with_io_strategy_with_loader{}(ev) &&
           io_strategy_requires_staging_storage{}(ev) &&
           io_load_batch_span_ready{}(ev) &&
           read_copy_storage_ready{}(ev);
  }
};

struct tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_with_io_strategy_with_loader{}(ev) &&
           io_strategy_requires_staging_storage{}(ev) &&
           io_load_batch_span_ready{}(ev) &&
           read_copy_storage_missing{}(ev);
  }
};

struct tensor_plan_done_with_read_copy_strategy_with_loader_and_storage_ready {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready{}(
        ev);
  }
};

struct
    tensor_plan_done_with_read_copy_strategy_with_loader_and_storage_missing {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return
        tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing{}(
            ev);
  }
};

struct tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_with_io_strategy_with_loader{}(ev) &&
           io_load_batch_span_missing{}(ev);
  }
};

struct io_load_done_all {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.io_events != nullptr && ev.io_events->load_done.raised &&
           !ev.io_events->load_error.raised &&
           ev.io_events->load_done.done_count ==
               ev.io_events->load_done.expected_count;
  }
};

struct io_load_error_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.io_events != nullptr && ev.io_events->load_error.raised;
  }
};

inline bool io_load_error_is(const event::load_runtime &ev,
                             const emel::error::type expected) noexcept {
  return io_load_error_raised{}(ev) && ev.io_events->load_error.err == expected;
}

struct io_load_error_invalid_request {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_load_error_is(
        ev, emel::error::cast(emel::io::loader::error::invalid_request));
  }
};

struct io_load_error_strategy_unavailable {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_load_error_is(
               ev, emel::error::cast(
                       emel::io::loader::error::unsupported_strategy)) ||
           io_load_error_is(
               ev, emel::error::cast(emel::io::loader::error::unavailable));
  }
};

struct io_load_error_internal {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_load_error_is(
        ev, emel::error::cast(emel::io::loader::error::internal_error));
  }
};

struct io_load_error_untracked {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_load_error_is(
        ev, emel::error::cast(emel::io::loader::error::untracked));
  }
};

struct io_load_error_unclassified {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return io_load_error_raised{}(ev) && !io_load_error_invalid_request{}(ev) &&
           !io_load_error_strategy_unavailable{}(ev) &&
           !io_load_error_internal{}(ev) && !io_load_error_untracked{}(ev);
  }
};

struct io_load_unhandled {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !io_load_done_all{}(ev) && !io_load_error_raised{}(ev);
  }
};

struct tensor_plan_error_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.plan_error.raised;
  }
};

struct tensor_plan_unhandled {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !tensor_plan_done_raised{}(ev) && !tensor_plan_error_raised{}(ev);
  }
};

struct tensor_apply_done_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.apply_done.raised;
  }
};

struct tensor_apply_done_with_file_image {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_apply_done_raised{}(ev) && has_file_image{}(ev);
  }
};

struct tensor_apply_done_without_file_image {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_apply_done_raised{}(ev) && !has_file_image{}(ev);
  }
};

struct tensor_apply_error_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.tensor_events.apply_error.raised;
  }
};

struct tensor_apply_unhandled {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !tensor_apply_done_raised{}(ev) && !tensor_apply_error_raised{}(ev);
  }
};

struct can_map_layers {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.map_layers);
  }
};

struct cannot_map_layers {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !can_map_layers{}(ev);
  }
};

struct skip_validate_structure {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !ev.request.check_tensors;
  }
};

struct can_validate_structure {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.check_tensors &&
           static_cast<bool>(ev.request.validate_structure);
  }
};

struct cannot_validate_structure {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.check_tensors &&
           !static_cast<bool>(ev.request.validate_structure);
  }
};

struct skip_validate_architecture {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !ev.request.validate_architecture;
  }
};

struct can_validate_architecture {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.validate_architecture &&
           static_cast<bool>(ev.request.validate_architecture_impl);
  }
};

struct cannot_validate_architecture {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.validate_architecture &&
           !static_cast<bool>(ev.request.validate_architecture_impl);
  }
};

struct done_callback_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

} // namespace emel::model::loader::guard
