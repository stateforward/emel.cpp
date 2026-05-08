#pragma once

#include <cstddef>
#include <cstring>

#include "emel/io/staged_read/context.hpp"
#include "emel/io/staged_read/detail.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/events.hpp"

namespace emel::io::staged_read::action {

struct effect_begin_staged_window {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
  }
};

struct effect_begin_staged_window_batch {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.done_count = 0u;
    ev.status.bytes_committed = 0u;
    ev.status.failed_index = 0u;
  }
};

struct effect_mark_invalid_staging_contract {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_stage_contract);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_callbacks {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_callbacks);
    ev.status.ok = false;
  }
};

struct effect_mark_batch_invalid_callbacks {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_callbacks);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_target_window {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_target_window);
    ev.status.ok = false;
  }
};

struct effect_mark_batch_invalid_staging_contract {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_stage_contract);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_platform {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_platform);
    ev.status.ok = false;
  }
};

struct effect_mark_batch_unsupported_platform {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_platform);
    ev.status.ok = false;
  }
};

struct effect_mark_staged_validation_accepted {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
  }
};

struct effect_mark_null_source_span {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::null_source_span);
    ev.status.ok = false;
  }
};

struct effect_mark_source_span_size_mismatch {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::source_span_size_mismatch);
    ev.status.ok = false;
  }
};

struct effect_mark_insufficient_source_span {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::insufficient_source_span);
    ev.status.ok = false;
  }
};

struct effect_publish_staged_window_done_aligned {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    const auto &rq = ev.request.request;
    auto *destination = static_cast<unsigned char *>(rq.target_buffer);
    const auto *origin = static_cast<const unsigned char *>(rq.source_span);

    const uint64_t logical = rq.logical_byte_length;
    const uint64_t chunk = rq.stage_chunk_bytes;
    uint64_t progressed = 0u;
    while (progressed != logical) {
      std::memcpy(destination + progressed, origin + progressed,
                  static_cast<std::size_t>(chunk));
      progressed += chunk;
    }

    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
    ev.request.on_done(events::staged_window_done{
        .intent = ev.request,
        .target_buffer = rq.target_buffer,
        .bytes_committed = logical,
    });
  }
};

struct effect_publish_staged_window_done_remainder {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    const auto &rq = ev.request.request;
    auto *destination = static_cast<unsigned char *>(rq.target_buffer);
    const auto *origin = static_cast<const unsigned char *>(rq.source_span);

    const uint64_t logical = rq.logical_byte_length;
    const uint64_t chunk = rq.stage_chunk_bytes;
    const uint64_t tail_bytes = logical % chunk;
    const uint64_t covered = logical - tail_bytes;

    uint64_t progressed = 0u;
    while (progressed != covered) {
      std::memcpy(destination + progressed, origin + progressed,
                  static_cast<std::size_t>(chunk));
      progressed += chunk;
    }
    std::memcpy(destination + progressed, origin + progressed,
                static_cast<std::size_t>(tail_bytes));

    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
    ev.request.on_done(events::staged_window_done{
        .intent = ev.request,
        .target_buffer = rq.target_buffer,
        .bytes_committed = logical,
    });
  }
};

struct effect_publish_staged_window_error {
  void operator()(const detail::staged_window_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::staged_window_error{
        .intent = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_publish_staged_window_batch_done {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    uint32_t done_count = 0u;
    uint64_t bytes_committed = 0u;
    for (uint32_t index = 0u;
         index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
      const auto &tensor = ev.request.tensors[index];
      auto *destination = static_cast<unsigned char *>(tensor.target);
      const auto *source_bytes =
          static_cast<const unsigned char *>(tensor.source_buffer);
      const auto *origin = source_bytes + tensor.file_offset;
      const uint64_t logical = tensor.byte_size;
      const uint64_t chunk = tensor.byte_size;
      const uint64_t tail_bytes = logical % chunk;
      const uint64_t covered = logical - tail_bytes;

      uint64_t progressed = 0u;
      while (progressed != covered) {
        std::memcpy(destination + progressed, origin + progressed,
                    static_cast<std::size_t>(chunk));
        progressed += chunk;
      }
      std::memcpy(destination + progressed, origin + progressed,
                  static_cast<std::size_t>(tail_bytes));
      done_count += 1u;
      bytes_committed += logical;
    }
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = true;
    ev.status.done_count = done_count;
    ev.status.bytes_committed = bytes_committed;
    ev.request.on_done(events::staged_window_batch_done{
        .intent = ev.request,
        .done_count = done_count,
        .bytes_committed = bytes_committed,
    });
  }
};

struct effect_publish_staged_window_batch_error {
  void operator()(const detail::staged_window_batch_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::staged_window_batch_error{
        .intent = ev.request,
        .err = ev.status.err,
        .failed_index = ev.status.failed_index,
    });
  }
};

struct effect_record_staged_window_error {
  void operator()(const detail::staged_window_runtime &,
                  context &) const noexcept {}
};

struct effect_record_staged_window_batch_error {
  void operator()(const detail::staged_window_batch_runtime &,
                  context &) const noexcept {}
};

struct effect_record_staged_window_batch_done {
  void operator()(const detail::staged_window_batch_runtime &,
                  context &) const noexcept {}
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_staged_window effect_begin_staged_window{};
inline constexpr effect_begin_staged_window_batch
    effect_begin_staged_window_batch{};
inline constexpr effect_mark_invalid_staging_contract
    effect_mark_invalid_staging_contract{};
inline constexpr effect_mark_invalid_callbacks effect_mark_invalid_callbacks{};
inline constexpr effect_mark_batch_invalid_callbacks
    effect_mark_batch_invalid_callbacks{};
inline constexpr effect_mark_invalid_target_window
    effect_mark_invalid_target_window{};
inline constexpr effect_mark_batch_invalid_staging_contract
    effect_mark_batch_invalid_staging_contract{};
inline constexpr effect_mark_unsupported_platform
    effect_mark_unsupported_platform{};
inline constexpr effect_mark_batch_unsupported_platform
    effect_mark_batch_unsupported_platform{};
inline constexpr effect_mark_staged_validation_accepted
    effect_mark_staged_validation_accepted{};
inline constexpr effect_mark_null_source_span effect_mark_null_source_span{};
inline constexpr effect_mark_source_span_size_mismatch
    effect_mark_source_span_size_mismatch{};
inline constexpr effect_mark_insufficient_source_span
    effect_mark_insufficient_source_span{};
inline constexpr effect_publish_staged_window_done_aligned
    effect_publish_staged_window_done_aligned{};
inline constexpr effect_publish_staged_window_done_remainder
    effect_publish_staged_window_done_remainder{};
inline constexpr effect_publish_staged_window_error
    effect_publish_staged_window_error{};
inline constexpr effect_publish_staged_window_batch_done
    effect_publish_staged_window_batch_done{};
inline constexpr effect_publish_staged_window_batch_error
    effect_publish_staged_window_batch_error{};
inline constexpr effect_record_staged_window_error
    effect_record_staged_window_error{};
inline constexpr effect_record_staged_window_batch_error
    effect_record_staged_window_batch_error{};
inline constexpr effect_record_staged_window_batch_done
    effect_record_staged_window_batch_done{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::staged_read::action
