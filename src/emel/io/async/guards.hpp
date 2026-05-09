#pragma once

#include <cstdint>
#include <limits>

#include "emel/io/async/detail.hpp"

namespace emel::io::async::guard {

struct guard_load_window_callbacks_present {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return static_cast<bool>(ev.intent.on_done) &&
           static_cast<bool>(ev.intent.on_progress) &&
           static_cast<bool>(ev.intent.on_error);
  }
};

struct guard_load_window_callbacks_missing {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_load_window_callbacks_present{}(ev);
  }
};

struct guard_source_contract_valid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    const auto &storage = ev.intent.request.storage;
    return storage.logical_byte_length > 0u &&
           storage.progress_chunk_bytes > 0u &&
           storage.source_span != nullptr &&
           storage.file_offset <= (std::numeric_limits<uint64_t>::max() -
                                   storage.logical_byte_length) &&
           storage.source_span_bytes >= storage.file_offset &&
           (storage.source_span_bytes - storage.file_offset) >=
               storage.logical_byte_length;
  }
};

struct guard_source_contract_invalid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_source_contract_valid{}(ev);
  }
};

struct guard_target_window_valid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    const auto &storage = ev.intent.request.storage;
    return storage.target_buffer != nullptr &&
           storage.target_window_bytes >= storage.logical_byte_length;
  }
};

struct guard_target_window_invalid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_target_window_valid{}(ev);
  }
};

struct guard_progress_contract_valid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    const auto &storage = ev.intent.request.storage;
    const auto &progress = ev.intent.request.progress;
    return progress.bytes_committed <= storage.logical_byte_length;
  }
};

struct guard_progress_contract_invalid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_progress_contract_valid{}(ev);
  }
};

struct guard_cancel_requested {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return ev.intent.request.progress.cancel_requested;
  }
};

struct guard_cancel_absent {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_cancel_requested{}(ev);
  }
};

struct guard_scheduler_contract_valid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    const auto &storage = ev.intent.request.storage;
    const auto &progress = ev.intent.request.progress;
    const uint64_t remaining =
        storage.logical_byte_length - progress.bytes_committed;
    const uint64_t delta =
        remaining > storage.progress_chunk_bytes ? storage.progress_chunk_bytes
                                                 : remaining;
    return delta <= storage.scheduler_resource_bytes;
  }
};

struct guard_scheduler_contract_invalid {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_scheduler_contract_valid{}(ev);
  }
};

struct guard_partial_progress_ready {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    const auto &storage = ev.intent.request.storage;
    const auto &progress = ev.intent.request.progress;
    const uint64_t remaining =
        storage.logical_byte_length - progress.bytes_committed;
    return remaining > storage.progress_chunk_bytes;
  }
};

struct guard_terminal_progress_ready {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !guard_partial_progress_ready{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return static_cast<bool>(ev.intent.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::load_window_runtime &ev) const noexcept {
    return !static_cast<bool>(ev.intent.on_error);
  }
};

} // namespace emel::io::async::guard
