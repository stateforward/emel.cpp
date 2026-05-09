#pragma once

#include <cstddef>
#include <cstring>

#include "emel/io/async/context.hpp"
#include "emel/io/async/detail.hpp"
#include "emel/io/async/errors.hpp"
#include "emel/io/async/events.hpp"

namespace emel::io::async::action {

struct effect_begin_load_window {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_strategy {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_strategy);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_callbacks {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_callbacks);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_source_contract {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_source_contract);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_target_window {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_target_window);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_progress_contract {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_progress_contract);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_scheduler_contract {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_scheduler_contract);
    ev.status.ok = false;
  }
};

struct effect_mark_cancelled {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::cancelled);
    ev.status.ok = false;
  }
};

struct effect_publish_load_window_progress_done {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    const auto &storage = ev.intent.request.storage;
    auto &progress = ev.intent.request.progress;
    auto *destination = static_cast<unsigned char *>(storage.target_buffer);
    const auto *source =
        static_cast<const unsigned char *>(storage.source_span) +
        storage.file_offset;
    const uint64_t offset = progress.bytes_committed;
    const uint64_t delta = storage.progress_chunk_bytes;

    std::memcpy(destination + offset, source + offset,
                static_cast<std::size_t>(delta));
    progress.bytes_committed = offset + delta;
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
    ev.intent.on_progress(events::load_window_progress_done{
        .intent = ev.intent,
        .target_buffer = storage.target_buffer,
        .bytes_committed = progress.bytes_committed,
        .bytes_delta = delta,
    });
  }
};

struct effect_publish_load_window_done {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    const auto &storage = ev.intent.request.storage;
    auto &progress = ev.intent.request.progress;
    auto *destination = static_cast<unsigned char *>(storage.target_buffer);
    const auto *source =
        static_cast<const unsigned char *>(storage.source_span) +
        storage.file_offset;
    const uint64_t offset = progress.bytes_committed;
    const uint64_t delta = storage.logical_byte_length - offset;

    std::memcpy(destination + offset, source + offset,
                static_cast<std::size_t>(delta));
    progress.bytes_committed = storage.logical_byte_length;
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
    ev.intent.on_done(events::load_window_done{
        .intent = ev.intent,
        .target_buffer = storage.target_buffer,
        .bytes_committed = progress.bytes_committed,
    });
  }
};

struct effect_publish_load_window_error {
  void operator()(const detail::load_window_runtime &ev,
                  context &) const noexcept {
    ev.intent.on_error(events::load_window_error{
        .intent = ev.intent,
        .err = ev.status.err,
    });
  }
};

struct effect_record_load_window_error {
  void operator()(const detail::load_window_runtime &,
                  context &) const noexcept {}
};

struct effect_on_unexpected {
  template <class event>
  void operator()(const event &, context &) const noexcept {}
};

} // namespace emel::io::async::action
