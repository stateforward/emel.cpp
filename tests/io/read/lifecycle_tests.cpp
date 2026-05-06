#include <cstdint>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/machines.hpp"

namespace {

struct read_owner_state {
  bool done = false;
  bool error = false;
  uint64_t bytes_copied = 0u;
  void *target_buffer = nullptr;
  emel::error::type err = emel::error::cast(emel::io::read::error::none);
};

struct batch_owner_state {
  bool done = false;
  bool error = false;
  uint32_t done_count = 0u;
  uint64_t bytes_copied = 0u;
  uint32_t failed_index = 0u;
  emel::error::type err = emel::error::cast(emel::io::read::error::none);
};

void on_read_done(void *object,
                  const emel::io::read::events::read_tensor_done &ev) noexcept {
  auto *owner = static_cast<read_owner_state *>(object);
  owner->done = true;
  owner->bytes_copied = ev.bytes_copied;
  owner->target_buffer = ev.target_buffer;
}

void on_read_error(
    void *object,
    const emel::io::read::events::read_tensor_error &ev) noexcept {
  auto *owner = static_cast<read_owner_state *>(object);
  owner->error = true;
  owner->err = ev.err;
}

void on_batch_done(
    void *object,
    const emel::io::read::events::read_tensor_batch_done &ev) noexcept {
  auto *owner = static_cast<batch_owner_state *>(object);
  owner->done = true;
  owner->done_count = ev.done_count;
  owner->bytes_copied = ev.bytes_copied;
}

void on_batch_error(
    void *object,
    const emel::io::read::events::read_tensor_batch_error &ev) noexcept {
  auto *owner = static_cast<batch_owner_state *>(object);
  owner->error = true;
  owner->err = ev.err;
  owner->failed_index = ev.failed_index;
}

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream input{path};
  REQUIRE(input.good());
  return std::string{std::istreambuf_iterator<char>{input},
                     std::istreambuf_iterator<char>{}};
}

emel::io::read::event::read_tensor_request
make_request(std::string_view path, void *target_buffer,
             uint64_t target_buffer_bytes) {
  return {
      .tensor_id = 42,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = target_buffer_bytes,
      .file_path = path,
      .target_buffer = target_buffer,
      .target_buffer_bytes = target_buffer_bytes,
  };
}

struct unrelated_event {};

} // namespace

TEST_CASE("io read exposes canonical machine aliases at component boundary") {
  emel::io::read::sm strategy{};
  emel::IoRead top_level_read{};

  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
  CHECK(
      top_level_read.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read copies requested bytes into caller-owned target buffer") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[4]{};
  constexpr char source[] = "abcdef";
  auto request = make_request("emel_io_read_success.bin", target, 3u);
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  request.file_offset = 2u;
  request.byte_size = 3u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  REQUIRE(strategy.process_event(read_request));
  REQUIRE(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_copied == 3u);
  CHECK(owner.target_buffer == target);
  CHECK(std::memcmp(target, "cde", 3u) == 0);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch copies caller owned tensor spans") {
  emel::io::read::sm strategy{};
  batch_owner_state owner{};
  uint8_t first_target[3]{};
  uint8_t second_target[4]{};
  constexpr char first_source[] = "abcdef";
  constexpr char second_source[] = "vwxyz";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 7,
          .file_index = 0u,
          .file_offset = 1u,
          .byte_size = 3u,
          .file_path = "emel_io_read_batch_first.bin",
          .source_buffer = first_source,
          .source_buffer_bytes = sizeof(first_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = first_target,
          .target_bytes = sizeof(first_target),
      },
      {
          .tensor_id = 8,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_second.bin",
          .source_buffer = second_source,
          .source_buffer_bytes = sizeof(second_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = second_target,
          .target_bytes = sizeof(second_target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};
  read_request.on_done = {&owner, on_batch_done};
  read_request.on_error = {&owner, on_batch_error};

  REQUIRE(strategy.process_event(read_request));
  REQUIRE(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.done_count == 2u);
  CHECK(owner.bytes_copied == 7u);
  CHECK(std::memcmp(first_target, "bcd", 3u) == 0);
  CHECK(std::memcmp(second_target, "vwxy", 4u) == 0);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch reports first failing span through explicit error") {
  emel::io::read::sm strategy{};
  batch_owner_state owner{};
  uint8_t first_target[3]{};
  uint8_t second_target[4]{};
  constexpr char first_source[] = "abcdef";
  constexpr char second_source[] = "xy";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 9,
          .file_index = 0u,
          .file_offset = 2u,
          .byte_size = 3u,
          .file_path = "emel_io_read_batch_ok.bin",
          .source_buffer = first_source,
          .source_buffer_bytes = sizeof(first_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = first_target,
          .target_bytes = sizeof(first_target),
      },
      {
          .tensor_id = 10,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_short.bin",
          .source_buffer = second_source,
          .source_buffer_bytes = sizeof(second_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = second_target,
          .target_bytes = sizeof(second_target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};
  read_request.on_done = {&owner, on_batch_done};
  read_request.on_error = {&owner, on_batch_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::short_read));
  CHECK(owner.failed_index == 1u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch reports invalid request span index") {
  emel::io::read::sm strategy{};
  batch_owner_state owner{};
  uint8_t first_target[3]{};
  uint8_t second_target[2]{};
  constexpr char first_source[] = "abcdef";
  constexpr char second_source[] = "wxyz";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 11,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 3u,
          .file_path = "emel_io_read_batch_valid.bin",
          .source_buffer = first_source,
          .source_buffer_bytes = sizeof(first_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = first_target,
          .target_bytes = sizeof(first_target),
      },
      {
          .tensor_id = 12,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_invalid.bin",
          .source_buffer = second_source,
          .source_buffer_bytes = sizeof(second_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = second_target,
          .target_bytes = sizeof(second_target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};
  read_request.on_done = {&owner, on_batch_done};
  read_request.on_error = {&owner, on_batch_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::invalid_request));
  CHECK(owner.failed_index == 1u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch reports unsupported resource span index") {
  emel::io::read::sm strategy{};
  batch_owner_state owner{};
  uint8_t first_target[3]{};
  uint8_t second_target[4]{};
  constexpr char first_source[] = "abcdef";
  constexpr char second_source[] = "wxyz";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 13,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 3u,
          .file_path = "emel_io_read_batch_resource_valid.bin",
          .source_buffer = first_source,
          .source_buffer_bytes = sizeof(first_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = first_target,
          .target_bytes = sizeof(first_target),
      },
      {
          .tensor_id = 14,
          .file_index = emel::io::read::k_max_file_index + 1u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_resource_bad.bin",
          .source_buffer = second_source,
          .source_buffer_bytes = sizeof(second_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = second_target,
          .target_bytes = sizeof(second_target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};
  read_request.on_done = {&owner, on_batch_done};
  read_request.on_error = {&owner, on_batch_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::unsupported_resource));
  CHECK(owner.failed_index == 1u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch reports source open and seek span indexes") {
  emel::io::read::sm strategy{};
  batch_owner_state open_owner{};
  batch_owner_state seek_owner{};
  uint8_t target[4]{};
  constexpr char source[] = "wxyz";
  const emel::io::event::tensor_load_span open_tensors[] = {
      {
          .tensor_id = 15,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_open.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .source_error =
              emel::error::cast(emel::io::read::error::file_open_failed),
          .target = target,
          .target_bytes = sizeof(target),
      },
  };
  const emel::io::event::tensor_load_span seek_tensors[] = {
      {
          .tensor_id = 16,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_seek.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .source_error =
              emel::error::cast(emel::io::read::error::file_seek_failed),
          .target = target,
          .target_bytes = sizeof(target),
      },
  };
  emel::io::read::event::read_tensor_batch open_request{open_tensors};
  open_request.on_done = {&open_owner, on_batch_done};
  open_request.on_error = {&open_owner, on_batch_error};
  emel::io::read::event::read_tensor_batch seek_request{seek_tensors};
  seek_request.on_done = {&seek_owner, on_batch_done};
  seek_request.on_error = {&seek_owner, on_batch_error};

  CHECK_FALSE(strategy.process_event(open_request));
  REQUIRE(open_owner.error);
  CHECK(open_owner.err ==
        emel::error::cast(emel::io::read::error::file_open_failed));
  CHECK(open_owner.failed_index == 0u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));

  CHECK_FALSE(strategy.process_event(seek_request));
  REQUIRE(seek_owner.error);
  CHECK(seek_owner.err ==
        emel::error::cast(emel::io::read::error::file_seek_failed));
  CHECK(seek_owner.failed_index == 0u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch reports file read span index") {
  emel::io::read::sm strategy{};
  batch_owner_state owner{};
  uint8_t first_target[3]{};
  uint8_t second_target[4]{};
  constexpr char first_source[] = "abcdef";
  constexpr char second_source[] = "wxyz";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 17,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 3u,
          .file_path = "emel_io_read_batch_read_valid.bin",
          .source_buffer = first_source,
          .source_buffer_bytes = sizeof(first_source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = first_target,
          .target_bytes = sizeof(first_target),
      },
      {
          .tensor_id = 18,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 4u,
          .file_path = "emel_io_read_batch_read_bad.bin",
          .source_buffer = second_source,
          .source_buffer_bytes = sizeof(second_source) - 1u,
          .source_error =
              emel::error::cast(emel::io::read::error::file_read_failed),
          .target = second_target,
          .target_bytes = sizeof(second_target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};
  read_request.on_done = {&owner, on_batch_done};
  read_request.on_error = {&owner, on_batch_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::file_read_failed));
  CHECK(owner.failed_index == 1u);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read batch fails closed without error callback") {
  emel::io::read::sm strategy{};
  uint8_t target[1]{};
  constexpr char source[] = "x";
  const emel::io::event::tensor_load_span tensors[] = {
      {
          .tensor_id = 19,
          .file_index = 0u,
          .file_offset = 0u,
          .byte_size = 2u,
          .file_path = "emel_io_read_batch_no_error.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .source_error = emel::error::cast(emel::io::read::error::none),
          .target = target,
          .target_bytes = sizeof(target),
      },
  };
  emel::io::read::event::read_tensor_batch read_request{tensors};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read can capture same-RTC result without public callbacks") {
  emel::io::read::sm strategy{};
  uint8_t target[4]{};
  constexpr char source[] = "abcdef";
  auto request = make_request("emel_io_read_result.bin", target, 4u);
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  request.file_offset = 1u;
  request.byte_size = 4u;
  emel::io::read::event::read_tensor read_request{request};
  emel::io::read::events::read_tensor_result result{};

  REQUIRE(strategy.process_event(read_request, result));
  CHECK(result.accepted);
  CHECK(result.ok);
  CHECK(result.err == emel::error::cast(emel::io::read::error::none));
  CHECK(result.bytes_copied == 4u);
  CHECK(result.target_buffer == target);
  CHECK(std::memcmp(target, "bcde", 4u) == 0);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read can capture same-RTC error without public callbacks") {
  emel::io::read::sm strategy{};
  uint8_t target[4]{};
  auto request = make_request("emel_io_read_result_error.bin", target, 4u);
  emel::io::read::event::read_tensor read_request{request};
  emel::io::read::events::read_tensor_result result{};

  CHECK_FALSE(strategy.process_event(read_request, result));
  CHECK(result.accepted);
  CHECK_FALSE(result.ok);
  CHECK(result.err ==
        emel::error::cast(emel::io::read::error::file_open_failed));
  CHECK(result.target_buffer == target);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read reports file open failures deterministically") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  const auto request = make_request("/tmp/emel_io_read_missing.bin", target,
                                    static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::file_open_failed));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read reports file seek failures deterministically") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  constexpr char source[] = "abcdefgh";
  auto request = make_request("emel_io_read_seek_failed.bin", target,
                              static_cast<uint64_t>(sizeof(target)));
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  request.source_error =
      emel::error::cast(emel::io::read::error::file_seek_failed);
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::file_seek_failed));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read reports file read failures deterministically") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  constexpr char source[] = "abcdefgh";
  auto request = make_request("emel_io_read_failed.bin", target,
                              static_cast<uint64_t>(sizeof(target)));
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  request.source_error =
      emel::error::cast(emel::io::read::error::file_read_failed);
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::file_read_failed));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read fails closed without an error callback") {
  emel::io::read::sm strategy{};
  uint8_t target[8]{};
  const auto request = make_request("/tmp/emel_io_read_no_callback.bin", target,
                                    static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE(
    "io read rejects invalid request preconditions before platform gate") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  auto request = make_request("/tmp/emel_io_read_invalid.bin", target,
                              static_cast<uint64_t>(sizeof(target)));
  request.byte_size = 0u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects invalid file path preconditions") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  const char path_with_nul[] = {'/', 't', 'm', 'p', '\0', 'x'};
  const auto request =
      make_request(std::string_view{path_with_nul, sizeof(path_with_nul)},
                   target, static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects unsupported file and length preconditions") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  auto request = make_request("/tmp/emel_io_read_file_index.bin", target,
                              static_cast<uint64_t>(sizeof(target)));
  request.file_index = emel::io::read::k_max_file_index + 1u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects oversized length preconditions") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  auto request = make_request("/tmp/emel_io_read_length.bin", target,
                              emel::io::read::k_max_read_bytes + 1u);
  request.byte_size = emel::io::read::k_max_read_bytes + 1u;
  request.target_buffer_bytes = emel::io::read::k_max_read_bytes + 1u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects unsupported layout preconditions") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  auto request = make_request("/tmp/emel_io_read_layout.bin", target,
                              static_cast<uint64_t>(sizeof(target)));
  request.file_offset = static_cast<uint64_t>(-4);
  request.byte_size = 8u;
  request.target_buffer_bytes = 8u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects invalid target-buffer preconditions") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[4]{};
  auto request = make_request("/tmp/emel_io_read_target.bin", target, 4u);
  request.target_buffer_bytes = 2u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read reports short reads deterministically") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  constexpr char source[] = "abc";
  auto request = make_request("emel_io_read_short.bin", target, sizeof(target));
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::read::error::short_read));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read recovers to ready after fail-closed dispatches") {
  emel::io::read::sm strategy{};
  read_owner_state first{};
  read_owner_state second{};
  uint8_t first_target[8]{};
  uint8_t second_target[8]{};
  const auto first_request =
      make_request("/tmp/emel_io_read_first.bin", first_target,
                   static_cast<uint64_t>(sizeof(first_target)));
  const auto second_request =
      make_request("/tmp/emel_io_read_second.bin", second_target,
                   static_cast<uint64_t>(sizeof(second_target)));
  emel::io::read::event::read_tensor first_read{first_request};
  emel::io::read::event::read_tensor second_read{second_request};
  first_read.on_error = {&first, on_read_error};
  second_read.on_error = {&second, on_read_error};

  CHECK_FALSE(strategy.process_event(first_read));
  CHECK(first.error);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
  CHECK_FALSE(strategy.process_event(second_read));
  CHECK(second.error);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read reports state_ready via visit_current_states") {
  emel::io::read::sm strategy{};
  uint8_t target[4]{};
  const auto request = make_request("/tmp/emel_io_read_visit.bin", target,
                                    static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};
  CHECK_FALSE(strategy.process_event(read_request));

  std::size_t visited_states = 0;
  bool saw_ready = false;
  strategy.visit_current_states([&](auto state) noexcept {
    ++visited_states;
    using state_t = typename decltype(state)::type;
    if constexpr (std::is_same_v<state_t, emel::io::read::state_ready>) {
      saw_ready = true;
    }
  });
  CHECK(visited_states == 1);
  CHECK(saw_ready);
}

TEST_CASE("io read handles unexpected events deterministically") {
  emel::io::read::sm strategy{};
  CHECK(strategy.process_event(unrelated_event{}));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read component contains no mmap or async strategy behavior") {
  const auto root = repo_root();
  const std::filesystem::path component = root / "src/emel/io/read";
  const std::string combined = read_text_file(component / "context.hpp") +
                               read_text_file(component / "detail.hpp") +
                               read_text_file(component / "errors.hpp") +
                               read_text_file(component / "events.hpp") +
                               read_text_file(component / "guards.hpp") +
                               read_text_file(component / "actions.hpp") +
                               read_text_file(component / "sm.hpp");

  const std::string_view source{combined};
  CHECK(source.find("mmap") == std::string_view::npos);
  CHECK(source.find("MapViewOfFile") == std::string_view::npos);
  CHECK(source.find("async") == std::string_view::npos);
  CHECK(source.find("staged") == std::string_view::npos);
  CHECK(source.find("chunked") == std::string_view::npos);
  CHECK(source.find("::open(") == std::string_view::npos);
  CHECK(source.find("::read(") == std::string_view::npos);
  CHECK(source.find("::lseek(") == std::string_view::npos);
  CHECK(source.find("::close(") == std::string_view::npos);
  CHECK(source.find("ReadFile") == std::string_view::npos);
  CHECK(source.find("CreateFile") == std::string_view::npos);
}
