#include <cstdint>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstring>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/error/error.hpp"
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

void write_text_file(const std::filesystem::path &path, std::string_view text) {
  std::ofstream output{path, std::ios::binary};
  REQUIRE(output.good());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  REQUIRE(output.good());
}

emel::io::read::event::read_tensor_request make_request(
    std::string_view path,
    void *target_buffer,
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
  const auto path = repo_root() / "build" / "emel_io_read_success.bin";
  write_text_file(path, "abcdef");
  const std::string path_text = path.string();
  auto request = make_request(path_text, target, 3u);
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

TEST_CASE("io read fails closed without an error callback") {
  emel::io::read::sm strategy{};
  uint8_t target[8]{};
  const auto request = make_request("/tmp/emel_io_read_no_callback.bin", target,
                                    static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}

TEST_CASE("io read rejects invalid request preconditions before platform gate") {
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

TEST_CASE("io read platform guard and unsupported action are explicit") {
  emel::io::read::action::context ctx{};
  emel::io::read::detail::read_attempt_status status{};
  uint8_t target[4]{};
  const auto request = make_request("/tmp/emel_io_read_platform.bin", target, 4u);
  emel::io::read::event::read_tensor read_request{request};
  emel::io::read::detail::read_tensor_runtime runtime{read_request, status};

  CHECK(emel::io::read::guard::platform_read_supported{}(runtime, ctx) !=
        emel::io::read::guard::platform_read_unsupported{}(runtime, ctx));
  emel::io::read::action::effect_mark_unsupported_platform(runtime, ctx);
  CHECK(status.err ==
        emel::error::cast(emel::io::read::error::unsupported_platform));
  CHECK_FALSE(status.ok);
}

TEST_CASE("io read reports short reads deterministically") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[8]{};
  const auto path = repo_root() / "build" / "emel_io_read_short.bin";
  write_text_file(path, "abc");
  const std::string path_text = path.string();
  const auto request = make_request(path_text, target, sizeof(target));
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
  const std::string combined =
      read_text_file(component / "actions.cpp") +
      read_text_file(component / "context.hpp") +
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
}
