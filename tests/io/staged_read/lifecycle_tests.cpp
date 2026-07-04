#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <doctest/doctest.h>

#include "emel/error/error.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/events.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/machines.hpp"
#include <stateforward/sml.hpp>

namespace {

struct staged_owner_state {
  bool done = false;
  bool error = false;
  int done_count = 0;
  int error_count = 0;
  emel::error::type err = emel::error::cast(emel::io::staged_read::error::none);
  void *target_buffer = nullptr;
  uint64_t bytes_committed = 0u;
  uint32_t failed_index = 0u;
};

struct staged_runtime_trace {
  bool callback_observed = false;
  const emel::io::staged_read::event::staged_window_request *request_seen =
      nullptr;
  const emel::io::staged_read::event::staged_window *intent_seen = nullptr;
};

void on_staged_done(
    void *object,
    const emel::io::staged_read::events::staged_window_done &ev) noexcept {
  auto *owner = static_cast<staged_owner_state *>(object);
  owner->done = true;
  owner->done_count++;
  owner->target_buffer = ev.target_buffer;
  owner->bytes_committed = ev.bytes_committed;
}

void on_staged_error(
    void *object,
    const emel::io::staged_read::events::staged_window_error &ev) noexcept {
  auto *owner = static_cast<staged_owner_state *>(object);
  owner->error = true;
  owner->error_count++;
  owner->err = ev.err;
}

void on_traced_done(
    void *object,
    const emel::io::staged_read::events::staged_window_done &ev) noexcept {
  auto *trace = static_cast<staged_runtime_trace *>(object);
  trace->callback_observed = true;
  trace->request_seen = &ev.intent.request;
  trace->intent_seen = &ev.intent;
}

struct staged_done_capture {
  bool done_observed = false;
  void *done_target = nullptr;
  uint64_t committed = 0u;
};

void on_captured_done(
    void *object,
    const emel::io::staged_read::events::staged_window_done &ev) noexcept {
  auto *capture = static_cast<staged_done_capture *>(object);
  capture->done_observed = true;
  capture->done_target = ev.target_buffer;
  capture->committed = ev.bytes_committed;
}

void on_ignore_error(
    void *,
    const emel::io::staged_read::events::staged_window_error &) noexcept {}

void on_staged_batch_done(
    void *object, const emel::io::staged_read::events::staged_window_batch_done
                      &ev) noexcept {
  auto *owner = static_cast<staged_owner_state *>(object);
  owner->done = true;
  owner->done_count = static_cast<int>(ev.done_count);
  owner->bytes_committed = ev.bytes_committed;
}

void on_staged_batch_error(
    void *object, const emel::io::staged_read::events::staged_window_batch_error
                      &ev) noexcept {
  auto *owner = static_cast<staged_owner_state *>(object);
  owner->error = true;
  owner->error_count++;
  owner->err = ev.err;
  owner->failed_index = ev.failed_index;
}

struct unrelated_event {};

} // namespace

// STG-07: context retains no dispatch-local request payload. The only member
// is the construction-time platform capability (persistent actor state), so
// the context stays trivially copyable and pointer-free at one flag wide.
static_assert(
    std::is_trivially_copyable_v<emel::io::staged_read::action::context> &&
        sizeof(emel::io::staged_read::action::context) == sizeof(bool),
    "staged_read::context must hold only the platform capability (STG-07)");

TEST_CASE("io staged_read default construction exposes state_ready") {
  emel::IoStagedRead machine{};
  REQUIRE(
      machine.is(stateforward::sml::state<emel::io::staged_read::state_ready>));
}

TEST_CASE("io staged_window per-attempt payload stays on same-RTC event stack "
          "(STG-07/LIFE-02)") {
  emel::IoStagedRead machine{};
  staged_runtime_trace trace{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> target{};
  std::array<unsigned char, 32> source{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = source.size();
  rq.stage_chunk_bytes = 8u;
  rq.source_span = source.data();
  rq.source_span_bytes = source.size();
  rq.target_buffer = target.data();
  rq.target_window_bytes = target.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&trace, on_traced_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE(machine.process_event(intent));
  REQUIRE(trace.callback_observed);
  REQUIRE(trace.request_seen == &rq);
  REQUIRE(trace.intent_seen == &intent);
}

TEST_CASE("io staged_window done event publishes caller-owned target only "
          "(SNR-01)") {
  emel::IoStagedRead machine{};
  staged_done_capture capture{};
  std::array<unsigned char, 64> target{};
  std::array<unsigned char, 64> source{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = source.size();
  rq.stage_chunk_bytes = 16u;
  rq.source_span = source.data();
  rq.source_span_bytes = source.size();
  rq.target_buffer = target.data();
  rq.target_window_bytes = target.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&capture, on_captured_done};
  intent.on_error = {nullptr, on_ignore_error};

  REQUIRE(machine.process_event(intent));
  REQUIRE(capture.done_observed);
  REQUIRE(capture.done_target == target.data());
  REQUIRE(capture.committed == rq.logical_byte_length);
}

TEST_CASE("io staged_window rejects logical span larger than declared target "
          "window (no overflow)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  constexpr unsigned k_window = 32u;
  constexpr unsigned k_canary_slots = 32u;
  std::array<unsigned char, k_window + k_canary_slots> backing{};
  for (auto &byte : backing) {
    byte = 0xEEu;
  }
  std::array<unsigned char, 48> src{};

  for (unsigned i = 0; i < src.size(); ++i) {
    src[i] = static_cast<unsigned char>(0x55u ^ i);
  }

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 48u;
  rq.stage_chunk_bytes = 16u;
  rq.source_span = src.data();
  rq.source_span_bytes = src.size();
  rq.target_buffer = backing.data();
  rq.target_window_bytes = k_window;

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE(rq.target_window_bytes >= rq.stage_chunk_bytes);
  REQUIRE(rq.target_window_bytes < rq.logical_byte_length);
  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE_FALSE(owner.done);
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_target_window));
  for (unsigned i = k_window; i < backing.size(); ++i) {
    REQUIRE(backing[i] == 0xEEu);
  }
}

TEST_CASE(
    "io staged_window copies full logical span through fixed chunk tiling "
    "(STG-04/06)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 128> buf{};
  std::array<unsigned char, 48> src{};

  for (unsigned i = 0; i < src.size(); ++i) {
    src[i] = static_cast<unsigned char>(0xA0 + i);
  }

  emel::io::staged_read::event::staged_window_request rq{};
  rq.file_offset = 0u;
  rq.logical_byte_length = 48u;
  rq.stage_chunk_bytes = 16u;
  rq.source_span = src.data();
  rq.source_span_bytes = src.size();
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE(machine.process_event(intent));
  REQUIRE(owner.done);
  REQUIRE_FALSE(owner.error);
  REQUIRE(owner.done_count == 1);
  REQUIRE(owner.error_count == 0);
  REQUIRE(owner.bytes_committed == rq.logical_byte_length);
  REQUIRE(owner.target_buffer == buf.data());
  REQUIRE(std::memcmp(buf.data(), src.data(),
                      static_cast<std::size_t>(rq.logical_byte_length)) == 0);
  REQUIRE(
      machine.is(stateforward::sml::state<emel::io::staged_read::state_ready>));
}

TEST_CASE("io staged_window copies logical span with non-divisible remainder "
          "(STG-04/STG-05)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 64> buf{};
  std::array<unsigned char, 48> src{};
  constexpr uint64_t logical = 48u;
  constexpr uint64_t chunk = 17u;

  static_assert(chunk <= logical && logical % chunk != 0u, "test shape");

  for (unsigned i = 0; i < src.size(); ++i) {
    src[i] = static_cast<unsigned char>(static_cast<unsigned>(i ^ 7u));
  }

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = logical;
  rq.stage_chunk_bytes = chunk;
  rq.source_span = src.data();
  rq.source_span_bytes = logical;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE(machine.process_event(intent));
  REQUIRE(owner.done);
  REQUIRE_FALSE(owner.error);
  REQUIRE(owner.done_count == 1);
  REQUIRE(owner.error_count == 0);
  REQUIRE(owner.bytes_committed == logical);
  REQUIRE(std::memcmp(buf.data(), src.data(),
                      static_cast<std::size_t>(logical)) == 0);
}

TEST_CASE(
    "io staged_window rejects mismatched caller source_span_bytes (staging)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};
  std::array<unsigned char, 32> src{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 16u;
  rq.stage_chunk_bytes = 16u;
  rq.source_span = src.data();
  rq.source_span_bytes = 31u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(owner.err ==
          emel::error::cast(
              emel::io::staged_read::error::source_span_size_mismatch));
}

TEST_CASE("io staged_window rejects absent source_span after platform guard") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 8u;
  rq.stage_chunk_bytes = 8u;
  rq.source_span = nullptr;
  rq.source_span_bytes = 0u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(owner.err ==
          emel::error::cast(emel::io::staged_read::error::null_source_span));
}

TEST_CASE("io staged_window rejects source_span shorter than logical span") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};
  std::array<unsigned char, 32> src{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 16u;
  rq.stage_chunk_bytes = 8u;
  rq.source_span = src.data();
  rq.source_span_bytes = 8u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(owner.err ==
          emel::error::cast(
              emel::io::staged_read::error::insufficient_source_span));
}

TEST_CASE(
    "io staged_window rejects missing callbacks as invalid_callbacks path") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 8u;
  rq.stage_chunk_bytes = 8u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(owner.err ==
          emel::error::cast(emel::io::staged_read::error::invalid_callbacks));
  REQUIRE(
      machine.is(stateforward::sml::state<emel::io::staged_read::state_ready>));
}

TEST_CASE("io staged_window rejects zero logical span (STG-02)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 8> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 0u;
  rq.stage_chunk_bytes = 8u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_stage_contract));
}

TEST_CASE("io staged_window rejects null target buffer (STG-03)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 32u;
  rq.stage_chunk_bytes = 16u;
  rq.target_buffer = nullptr;
  rq.target_window_bytes = 64u;

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_target_window));
}

TEST_CASE("io staged_window rejects undersized caller target window versus "
          "stage slab "
          "(STG-03)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 8> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 64u;
  rq.stage_chunk_bytes = 32u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = 15u;

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_target_window));
}

TEST_CASE(
    "io staged_window rejects stage chunk larger than logical span (STG-02)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 64> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 24u;
  rq.stage_chunk_bytes = 64u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE_FALSE(owner.done);
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_stage_contract));
}

TEST_CASE("io staged_window rejects uint64 span overflow at file_offset + "
          "logical length") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.file_offset = ~0ull - 3u;
  rq.logical_byte_length = 8u;
  rq.stage_chunk_bytes = 8u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_stage_contract));
}

TEST_CASE("io staged_window rejects zero stage chunk bytes (STG-02)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 8> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 8u;
  rq.stage_chunk_bytes = 0u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE(owner.error);
}

TEST_CASE("io staged_window rejects target window smaller than stage chunk "
          "(STG-03)") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 16u;
  rq.stage_chunk_bytes = 16u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = 8u;

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {&owner, on_staged_done};
  intent.on_error = {&owner, on_staged_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE_FALSE(owner.done);
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_target_window));
}

TEST_CASE("io staged_window records error without firing on_error when both "
          "callbacks missing") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  std::array<unsigned char, 32> buf{};

  emel::io::staged_read::event::staged_window_request rq{};
  rq.logical_byte_length = 8u;
  rq.stage_chunk_bytes = 8u;
  rq.target_buffer = buf.data();
  rq.target_window_bytes = buf.size();

  emel::io::staged_read::event::staged_window intent{rq};
  intent.on_done = {};
  intent.on_error = {};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE_FALSE(owner.done);
  REQUIRE_FALSE(owner.error);
}

TEST_CASE("io staged_window_batch reports first invalid tensor index") {
  emel::IoStagedRead machine{};
  staged_owner_state owner{};
  constexpr char source[] = "abcdefgh";
  std::array<char, 2> first_target{};
  std::array<char, 4> second_target{};
  const std::array<emel::io::event::tensor_load_span, 2> tensors{{
      {
          .tensor_id = 1,
          .file_offset = 0u,
          .byte_size = first_target.size(),
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = first_target.data(),
          .target_bytes = first_target.size(),
      },
      {
          .tensor_id = 2,
          .file_offset = 6u,
          .byte_size = second_target.size(),
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = second_target.data(),
          .target_bytes = second_target.size(),
      },
  }};

  emel::io::staged_read::event::staged_window_batch intent{tensors, 2u};
  intent.on_done = {&owner, on_staged_batch_done};
  intent.on_error = {&owner, on_staged_batch_error};

  REQUIRE_FALSE(machine.process_event(intent));
  REQUIRE_FALSE(owner.done);
  REQUIRE(owner.error);
  REQUIRE(
      owner.err ==
      emel::error::cast(emel::io::staged_read::error::invalid_stage_contract));
  REQUIRE(owner.failed_index == 1u);
}

TEST_CASE("io staged_read handles unexpected events deterministically") {
  emel::io::staged_read::sm strategy{};
  CHECK(strategy.process_event(unrelated_event{}));
  CHECK(strategy.is(
      stateforward::sml::state<emel::io::staged_read::state_ready>));
}
