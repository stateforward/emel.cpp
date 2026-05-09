#pragma once

#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/events.hpp"
#include "emel/io/staged_read/sm.hpp"

namespace emel::io::loader::action {

struct effect_begin_load_tensor {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.strategy_err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.bytes_copied = 0u;
    ev.ctx.buffer = nullptr;
  }
};

struct effect_begin_load_tensor_batch {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.strategy_err = emel::error::cast(error::none);
    ev.status.accepted = false;
    ev.status.ok = false;
    ev.status.done_count = 0u;
    ev.status.bytes_done = 0u;
    ev.status.failed_index = 0u;
  }
};

struct effect_mark_invalid_request {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    ev.ctx.ok = false;
  }
};

struct effect_mark_load_tensor_batch_invalid_request {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_strategy {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::unsupported_strategy);
    ev.ctx.ok = false;
  }
};

struct effect_mark_load_tensor_batch_unsupported_strategy {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_strategy);
    ev.status.ok = false;
  }
};

struct effect_publish_load_tensor_error {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::load_tensor_error{
        .request = ev.request,
        .err = ev.ctx.err,
        .strategy_err = ev.ctx.strategy_err,
    });
  }
};

struct effect_record_load_tensor_error {
  void operator()(const detail::load_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_load_tensor_batch_error {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::load_tensor_batch_error{
        .request = ev.request,
        .err = ev.status.err,
        .strategy_err = ev.status.strategy_err,
        .failed_index = ev.status.failed_index,
    });
  }
};

struct effect_record_load_tensor_batch_error {
  void operator()(const detail::load_tensor_batch_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_load_tensor_done {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::load_tensor_done{
        .request = ev.request,
        .strategy = ev.request.policy.strategy,
        .buffer = ev.ctx.buffer,
        .buffer_bytes = ev.ctx.bytes_copied,
    });
  }
};

struct effect_publish_load_tensor_batch_done {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::load_tensor_batch_done{
        .request = ev.request,
        .strategy = ev.request.policy.strategy,
        .done_count = ev.status.done_count,
        .bytes_done = ev.status.bytes_done,
    });
  }
};

struct effect_record_load_tensor_done {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = true;
  }
};

struct effect_record_load_tensor_batch_done {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
  }
};

struct effect_record_read_tensor_batch_failed {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unavailable);
    ev.status.ok = false;
  }
};

namespace read_callbacks {

inline void
on_read_done(void *object,
             const emel::io::read::events::read_tensor_done &ev) noexcept {
  auto *status = static_cast<detail::runtime_status *>(object);
  status->err = emel::error::cast(error::none);
  status->strategy_err = emel::error::cast(emel::io::read::error::none);
  status->ok = true;
  status->bytes_copied = ev.bytes_copied;
  status->buffer = ev.target_buffer;
}

inline void
on_read_error(void *object,
              const emel::io::read::events::read_tensor_error &ev) noexcept {
  auto *status = static_cast<detail::runtime_status *>(object);
  status->err = emel::error::cast(error::unavailable);
  status->strategy_err = ev.err;
  status->ok = false;
}

} // namespace read_callbacks

namespace staged_read_callbacks {

inline void on_staged_read_done(
    void *object,
    const emel::io::staged_read::events::staged_window_done &ev) noexcept {
  auto *status = static_cast<detail::runtime_status *>(object);
  status->err = emel::error::cast(error::none);
  status->strategy_err = emel::error::cast(emel::io::staged_read::error::none);
  status->ok = true;
  status->bytes_copied = ev.bytes_committed;
  status->buffer = ev.target_buffer;
}

inline void on_staged_read_error(
    void *object,
    const emel::io::staged_read::events::staged_window_error &ev) noexcept {
  auto *status = static_cast<detail::runtime_status *>(object);
  status->err = emel::error::cast(error::unavailable);
  status->strategy_err = ev.err;
  status->ok = false;
}

} // namespace staged_read_callbacks

namespace read_batch_callbacks {

inline void on_read_batch_done(
    void *object,
    const emel::io::read::events::read_tensor_batch_done &ev) noexcept {
  auto *status = static_cast<detail::batch_runtime_status *>(object);
  status->err = emel::error::cast(error::none);
  status->strategy_err = emel::error::cast(emel::io::read::error::none);
  status->ok = true;
  status->done_count = ev.done_count;
  status->bytes_done = ev.bytes_copied;
}

inline void on_read_batch_error(
    void *object,
    const emel::io::read::events::read_tensor_batch_error &ev) noexcept {
  auto *status = static_cast<detail::batch_runtime_status *>(object);
  status->err = emel::error::cast(error::unavailable);
  status->strategy_err = ev.err;
  status->ok = false;
  status->failed_index = ev.failed_index;
}

} // namespace read_batch_callbacks

struct effect_dispatch_read_tensor {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::read::event::read_tensor_request request{
        .tensor_id = ev.request.tensor.tensor_id,
        .file_index = ev.request.tensor.file_index,
        .file_offset = ev.request.tensor.file_offset,
        .byte_size = ev.request.tensor.byte_size,
        .file_path = ev.request.tensor.file_path,
        .source_buffer = ev.request.tensor.source_buffer,
        .source_buffer_bytes = ev.request.tensor.source_buffer_bytes,
        .source_error = ev.request.tensor.source_error,
        .target_buffer = ev.request.tensor.target,
        .target_buffer_bytes = ev.request.tensor.target_bytes,
    };
    emel::io::read::event::read_tensor read{request};
    read.on_done = {static_cast<void *>(&ev.ctx), read_callbacks::on_read_done};
    read.on_error = {static_cast<void *>(&ev.ctx),
                     read_callbacks::on_read_error};
    static_cast<void>(ctx.io_read->process_event(read));
  }
};

struct effect_dispatch_staged_read_tensor {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &ctx) const noexcept {
    const auto &tensor = ev.request.tensor;
    const auto *source_bytes =
        static_cast<const uint8_t *>(tensor.source_buffer);
    const auto request = emel::io::staged_read::event::staged_window_request{
        .file_offset = tensor.file_offset,
        .logical_byte_length = tensor.byte_size,
        .stage_chunk_bytes = detail::compute_staged_chunk_bytes(
            ev.request.policy.staged_chunk_bytes, tensor.byte_size),
        .source_span = source_bytes + tensor.file_offset,
        .source_span_bytes = tensor.byte_size,
        .target_buffer = tensor.target,
        .target_window_bytes = tensor.byte_size,
    };
    emel::io::staged_read::event::staged_window staged_window{request};
    staged_window.on_done = {static_cast<void *>(&ev.ctx),
                             staged_read_callbacks::on_staged_read_done};
    staged_window.on_error = {static_cast<void *>(&ev.ctx),
                              staged_read_callbacks::on_staged_read_error};
    static_cast<void>(ctx.io_staged_read->process_event(staged_window));
  }
};

struct effect_dispatch_read_tensor_batch {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::read::event::read_tensor_batch read{ev.request.tensors};
    read.on_done = {static_cast<void *>(&ev.status),
                    read_batch_callbacks::on_read_batch_done};
    read.on_error = {static_cast<void *>(&ev.status),
                     read_batch_callbacks::on_read_batch_error};
    ev.status.accepted = ctx.io_read->process_event(read);
  }
};

struct effect_dispatch_staged_read_tensor_batch {
  void operator()(const detail::load_tensor_batch_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::staged_read::event::staged_window_batch batch{
        ev.request.tensors, ev.request.policy.staged_chunk_bytes};
    batch.on_done = {
        static_cast<void *>(&ev.status),
        [](void *object,
           const emel::io::staged_read::events::staged_window_batch_done
               &done) noexcept {
          auto *status = static_cast<detail::batch_runtime_status *>(object);
          status->accepted = true;
          status->ok = true;
          status->err = emel::error::cast(error::none);
          status->strategy_err =
              emel::error::cast(emel::io::staged_read::error::none);
          status->done_count = done.done_count;
          status->bytes_done = done.bytes_committed;
        }};
    batch.on_error = {
        static_cast<void *>(&ev.status),
        [](void *object,
           const emel::io::staged_read::events::staged_window_batch_error
               &err_ev) noexcept {
          auto *status = static_cast<detail::batch_runtime_status *>(object);
          status->accepted = false;
          status->ok = false;
          status->err = emel::error::cast(error::unavailable);
          status->strategy_err = err_ev.err;
          status->failed_index = err_ev.failed_index;
        }};
    ev.status.accepted = ctx.io_staged_read->process_event(batch);
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      ev.ctx.ok = false;
    } else if constexpr (requires { ev.status.err; }) {
      ev.status.err = emel::error::cast(error::internal_error);
      ev.status.ok = false;
      ev.status.accepted = false;
    }
  }
};

inline constexpr effect_begin_load_tensor effect_begin_load_tensor{};
inline constexpr effect_begin_load_tensor_batch
    effect_begin_load_tensor_batch{};
inline constexpr effect_mark_invalid_request effect_mark_invalid_request{};
inline constexpr effect_mark_load_tensor_batch_invalid_request
    effect_mark_load_tensor_batch_invalid_request{};
inline constexpr effect_mark_unsupported_strategy
    effect_mark_unsupported_strategy{};
inline constexpr effect_mark_load_tensor_batch_unsupported_strategy
    effect_mark_load_tensor_batch_unsupported_strategy{};
inline constexpr effect_publish_load_tensor_error
    effect_publish_load_tensor_error{};
inline constexpr effect_record_load_tensor_error
    effect_record_load_tensor_error{};
inline constexpr effect_publish_load_tensor_batch_error
    effect_publish_load_tensor_batch_error{};
inline constexpr effect_record_load_tensor_batch_error
    effect_record_load_tensor_batch_error{};
inline constexpr effect_publish_load_tensor_done
    effect_publish_load_tensor_done{};
inline constexpr effect_record_load_tensor_done
    effect_record_load_tensor_done{};
inline constexpr effect_publish_load_tensor_batch_done
    effect_publish_load_tensor_batch_done{};
inline constexpr effect_record_load_tensor_batch_done
    effect_record_load_tensor_batch_done{};
inline constexpr effect_record_read_tensor_batch_failed
    effect_record_read_tensor_batch_failed{};
inline constexpr effect_dispatch_read_tensor effect_dispatch_read_tensor{};
inline constexpr effect_dispatch_staged_read_tensor
    effect_dispatch_staged_read_tensor{};
inline constexpr effect_dispatch_read_tensor_batch
    effect_dispatch_read_tensor_batch{};
inline constexpr effect_dispatch_staged_read_tensor_batch
    effect_dispatch_staged_read_tensor_batch{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::loader::action
