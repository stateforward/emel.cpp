#include <cstdint>

#include <filesystem>
#include <fstream>
#include <iterator>
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

TEST_CASE("io read fails closed for accepted boundary requests") {
  emel::io::read::sm strategy{};
  read_owner_state owner{};
  uint8_t target[16]{};
  const auto request = make_request("/tmp/emel_io_read_boundary.bin", target,
                                    static_cast<uint64_t>(sizeof(target)));
  emel::io::read::event::read_tensor read_request{request};
  read_request.on_done = {&owner, on_read_done};
  read_request.on_error = {&owner, on_read_error};

  CHECK_FALSE(strategy.process_event(read_request));
  CHECK_FALSE(owner.done);
  REQUIRE(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::read::error::unsupported_platform));
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

TEST_CASE("io read boundary contains no concrete platform read calls") {
  const auto root = repo_root();
  const std::filesystem::path component = root / "src/emel/io/read";
  const std::string combined =
      read_text_file(component / "context.hpp") +
      read_text_file(component / "detail.hpp") +
      read_text_file(component / "errors.hpp") +
      read_text_file(component / "events.hpp") +
      read_text_file(component / "guards.hpp") +
      read_text_file(component / "actions.hpp") +
      read_text_file(component / "sm.hpp");

  const std::string_view source{combined};
  CHECK(source.find("pread") == std::string_view::npos);
  CHECK(source.find("read(") == std::string_view::npos);
  CHECK(source.find("lseek") == std::string_view::npos);
  CHECK(source.find("open(") == std::string_view::npos);
  CHECK(source.find("close(") == std::string_view::npos);
  CHECK(source.find("ReadFile") == std::string_view::npos);
  CHECK(source.find("CreateFileW") == std::string_view::npos);
  CHECK(source.find("ifstream") == std::string_view::npos);
  CHECK(source.find("fread") == std::string_view::npos);
  CHECK(source.find("fopen") == std::string_view::npos);
  CHECK(source.find("fseek") == std::string_view::npos);
  CHECK(source.find("fclose") == std::string_view::npos);
}
