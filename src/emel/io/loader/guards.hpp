#pragma once

#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/staged_read/errors.hpp"

namespace emel::io::loader::guard {

struct tensor_span_valid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.tensor.byte_size > 0u &&
           ev.request.tensor.target != nullptr &&
           ev.request.tensor.target_bytes >= ev.request.tensor.byte_size;
  }
};

struct tensor_span_invalid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !tensor_span_valid{}(ev, ctx);
  }
};

struct batch_span_valid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool valid = !ev.request.tensors.empty();
    for (uint32_t index = 0u;
         index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
      const auto &span = ev.request.tensors[index];
      valid = valid && span.byte_size > 0u && span.target != nullptr &&
              span.target_bytes >= span.byte_size;
    }
    return valid;
  }
};

struct batch_span_invalid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_span_valid{}(ev, ctx);
  }
};

struct strategy_none {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::none;
  }
};

struct strategy_mapped_file {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::mapped_file;
  }
};

struct strategy_read_copy {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::read_copy;
  }
};

struct strategy_read_copy_batch {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::read_copy;
  }
};

struct strategy_staged_read {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::staged_read;
  }
};

struct strategy_staged_read_batch {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::staged_read;
  }
};

struct strategy_cooperative_async {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy ==
           event::strategy_kind::cooperative_async;
  }
};

struct strategy_cooperative_async_batch {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.request.policy.strategy ==
           event::strategy_kind::cooperative_async;
  }
};

struct read_actor_present {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.io_read != nullptr;
  }
};

struct read_actor_absent {
  bool operator()(const action::context &ctx) const noexcept {
    return !read_actor_present{}(ctx);
  }
};

struct staged_read_actor_present {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.io_staged_read != nullptr;
  }
};

struct staged_read_actor_absent {
  bool operator()(const action::context &ctx) const noexcept {
    return !staged_read_actor_present{}(ctx);
  }
};

struct async_actor_present {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.io_async != nullptr;
  }
};

struct async_actor_absent {
  bool operator()(const action::context &ctx) const noexcept {
    return !async_actor_present{}(ctx);
  }
};

struct strategy_read_copy_with_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_read_copy{}(ev) && read_actor_present{}(ctx);
  }
};

struct strategy_read_copy_without_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_read_copy{}(ev) && read_actor_absent{}(ctx);
  }
};

struct strategy_staged_read_with_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read{}(ev) && staged_read_actor_present{}(ctx);
  }
};

struct strategy_staged_read_without_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read{}(ev) && staged_read_actor_absent{}(ctx);
  }
};

struct strategy_cooperative_async_with_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async{}(ev) && async_actor_present{}(ctx);
  }
};

struct strategy_cooperative_async_without_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async{}(ev) && async_actor_absent{}(ctx);
  }
};

struct strategy_read_copy_batch_with_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_read_copy_batch{}(ev) && read_actor_present{}(ctx);
  }
};

struct strategy_read_copy_batch_without_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_read_copy_batch{}(ev) && read_actor_absent{}(ctx);
  }
};

struct strategy_staged_read_batch_with_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_batch{}(ev) && staged_read_actor_present{}(ctx);
  }
};

struct strategy_staged_read_batch_without_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_batch{}(ev) && staged_read_actor_absent{}(ctx);
  }
};

struct strategy_cooperative_async_batch_with_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_batch{}(ev) && async_actor_present{}(ctx);
  }
};

struct strategy_cooperative_async_batch_without_actor {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_batch{}(ev) && async_actor_absent{}(ctx);
  }
};

struct staged_read_source_span_valid {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    const auto &tensor = ev.request.tensor;
    return tensor.source_buffer != nullptr &&
           tensor.source_buffer_bytes >= tensor.file_offset &&
           (tensor.source_buffer_bytes - tensor.file_offset) >=
               tensor.byte_size;
  }
};

struct staged_read_source_span_invalid {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !staged_read_source_span_valid{}(ev);
  }
};

struct strategy_staged_read_with_actor_and_source_span_valid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_with_actor{}(ev, ctx) &&
           staged_read_source_span_valid{}(ev);
  }
};

struct strategy_staged_read_with_actor_and_source_span_invalid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_with_actor{}(ev, ctx) &&
           staged_read_source_span_invalid{}(ev);
  }
};

struct strategy_cooperative_async_with_actor_and_source_span_valid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_with_actor{}(ev, ctx) &&
           staged_read_source_span_valid{}(ev) &&
           ev.request.async_progress != nullptr &&
           ev.request.async_progress->bytes_committed <=
               ev.request.tensor.byte_size;
  }
};

struct strategy_cooperative_async_with_actor_and_source_span_invalid {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_with_actor{}(ev, ctx) &&
           !strategy_cooperative_async_with_actor_and_source_span_valid{}(ev,
                                                                          ctx);
  }
};

struct staged_read_batch_source_span_valid {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    bool valid = true;
    for (uint32_t index = 0u;
         index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
      const auto &tensor = ev.request.tensors[index];
      valid =
          valid && tensor.source_buffer != nullptr &&
          tensor.source_buffer_bytes >= tensor.file_offset &&
          (tensor.source_buffer_bytes - tensor.file_offset) >= tensor.byte_size;
    }
    return valid;
  }
};

struct staged_read_batch_source_span_invalid {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !staged_read_batch_source_span_valid{}(ev);
  }
};

struct strategy_staged_read_batch_with_actor_and_source_span_valid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_batch_with_actor{}(ev, ctx) &&
           staged_read_batch_source_span_valid{}(ev);
  }
};

struct strategy_staged_read_batch_with_actor_and_source_span_invalid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_staged_read_batch_with_actor{}(ev, ctx) &&
           staged_read_batch_source_span_invalid{}(ev);
  }
};

struct strategy_cooperative_async_batch_with_actor_and_source_span_valid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_batch_with_actor{}(ev, ctx) &&
           staged_read_batch_source_span_valid{}(ev) &&
           ev.request.async_batch_progress != nullptr &&
           ev.request.async_progress.size() >= ev.request.tensors.size() &&
           ev.request.async_batch_progress->next_index <
               static_cast<uint32_t>(ev.request.tensors.size()) &&
           ev.request.async_batch_progress->done_count <=
               static_cast<uint32_t>(ev.request.tensors.size()) &&
           ev.request
                   .async_progress[ev.request.async_batch_progress->next_index]
                   .bytes_committed <=
               ev.request.tensors[ev.request.async_batch_progress->next_index]
                   .byte_size;
  }
};

struct strategy_cooperative_async_batch_with_actor_and_source_span_invalid {
  bool operator()(const detail::load_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_cooperative_async_batch_with_actor{}(ev, ctx) &&
           !strategy_cooperative_async_batch_with_actor_and_source_span_valid{}(
               ev, ctx);
  }
};

struct read_load_succeeded {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.ctx.ok;
  }
};

struct read_load_failed {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !read_load_succeeded{}(ev);
  }
};

struct async_load_progressed {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.ctx.accepted && ev.ctx.ok && ev.ctx.partial;
  }
};

struct async_load_succeeded {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.ctx.accepted && ev.ctx.ok && ev.ctx.done;
  }
};

struct async_load_failed {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !async_load_progressed{}(ev) && !async_load_succeeded{}(ev);
  }
};

struct read_batch_succeeded {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.status.accepted && ev.status.ok;
  }
};

struct read_batch_failed {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !ev.status.accepted || !ev.status.ok;
  }
};

struct async_batch_progressed {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.status.accepted && ev.status.ok &&
           ev.status.done_count <
               static_cast<uint32_t>(ev.request.tensors.size());
  }
};

struct async_batch_succeeded {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return ev.status.accepted && ev.status.ok &&
           ev.status.done_count ==
               static_cast<uint32_t>(ev.request.tensors.size());
  }
};

struct async_batch_failed {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !async_batch_progressed{}(ev) && !async_batch_succeeded{}(ev);
  }
};

struct strategy_external_buffer {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::external_buffer;
  }
};

struct done_callback_present {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct batch_done_callback_present {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct batch_done_callback_absent {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !batch_done_callback_present{}(ev);
  }
};

struct progress_callback_present {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_progress);
  }
};

struct progress_callback_absent {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !progress_callback_present{}(ev);
  }
};

struct batch_progress_callback_present {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_progress);
  }
};

struct batch_progress_callback_absent {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !batch_progress_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

struct batch_error_callback_present {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct batch_error_callback_absent {
  bool operator()(const detail::load_tensor_batch_runtime &ev) const noexcept {
    return !batch_error_callback_present{}(ev);
  }
};

} // namespace emel::io::loader::guard
