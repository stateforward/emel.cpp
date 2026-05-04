#include <cstdint>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <doctest/doctest.h>

#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/machines.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::io::loader::error::none);
  emel::io::loader::event::strategy_kind strategy =
      emel::io::loader::event::strategy_kind::none;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

void on_load_done(
    void *object,
    const emel::io::loader::events::load_tensor_done &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->strategy = ev.strategy;
  owner->buffer = ev.buffer;
  owner->buffer_bytes = ev.buffer_bytes;
}

void on_load_error(
    void *object,
    const emel::io::loader::events::load_tensor_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
}

void *fake_target(const uintptr_t value) {
  return reinterpret_cast<void *>(value);
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

} // namespace

TEST_CASE("io loader exposes canonical machine aliases") {
  emel::io::loader::sm loader{};
  emel::io::sm io_loader{};
  emel::IoLoader top_level_loader{};

  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
  CHECK(io_loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
  CHECK(top_level_loader.is(
      stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader rejects invalid tensor spans before strategy handling") {
  emel::io::loader::sm loader{};
  owner_state owner{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 7,
      .file_index = 2u,
      .file_offset = 128u,
      .byte_size = 0u,
      .target = fake_target(0xC000u),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::mapped_file,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};
  request.on_done = {&owner, on_load_done};
  request.on_error = {&owner, on_load_error};

  CHECK_FALSE(loader.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::invalid_request));
}

TEST_CASE("io loader fails closed for absent and explicit strategies") {
  emel::io::loader::sm loader{};
  owner_state owner{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 3,
      .file_index = 1u,
      .file_offset = 4096u,
      .byte_size = 64u,
      .target = fake_target(0xD000u),
  };
  const std::array strategies{
      emel::io::loader::event::strategy_kind::none,
      emel::io::loader::event::strategy_kind::mapped_file,
      emel::io::loader::event::strategy_kind::staged_read,
      emel::io::loader::event::strategy_kind::external_buffer,
  };

  for (const auto strategy : strategies) {
    CAPTURE(static_cast<uint32_t>(strategy));
    owner = {};
    const emel::io::loader::event::strategy_policy policy{strategy};
    emel::io::loader::event::load_tensor request{tensor, policy};
    request.on_done = {&owner, on_load_done};
    request.on_error = {&owner, on_load_error};

    CHECK_FALSE(loader.process_event(request));
    CHECK_FALSE(owner.done);
    CHECK(owner.error);
    CHECK(owner.err ==
          emel::error::cast(emel::io::loader::error::unsupported_strategy));
  }
}

TEST_CASE("io loader fails closed and recovers for unknown strategies") {
  emel::io::loader::sm loader{};
  owner_state owner{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 4,
      .file_index = 1u,
      .file_offset = 8192u,
      .byte_size = 128u,
      .target = fake_target(0xD800u),
  };
  const emel::io::loader::event::strategy_policy unknown_policy{
      static_cast<emel::io::loader::event::strategy_kind>(0xFFu),
  };
  emel::io::loader::event::load_tensor unknown_request{tensor,
                                                       unknown_policy};
  unknown_request.on_done = {&owner, on_load_done};
  unknown_request.on_error = {&owner, on_load_error};

  CHECK_FALSE(loader.process_event(unknown_request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::unsupported_strategy));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));

  owner = {};
  const emel::io::loader::event::strategy_policy explicit_policy{
      emel::io::loader::event::strategy_kind::staged_read,
  };
  emel::io::loader::event::load_tensor explicit_request{tensor,
                                                        explicit_policy};
  explicit_request.on_done = {&owner, on_load_done};
  explicit_request.on_error = {&owner, on_load_error};

  CHECK_FALSE(loader.process_event(explicit_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::unsupported_strategy));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader boundary has no concrete system IO strategy code") {
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "loader" / "actions.hpp");
  const std::string sm_source =
      read_text_file(repo_root() / "src" / "emel" / "io" / "loader" / "sm.hpp");

  CHECK(actions_source.find("mmap(") == std::string::npos);
  CHECK(actions_source.find("pread(") == std::string::npos);
  CHECK(actions_source.find("std::ifstream") == std::string::npos);
  CHECK(actions_source.find("CreateFileMapping") == std::string::npos);
  CHECK(sm_source.find("strategy_mapped_file") != std::string::npos);
  CHECK(sm_source.find("strategy_staged_read") != std::string::npos);
  CHECK(sm_source.find("strategy_external_buffer") != std::string::npos);
  CHECK(sm_source.find("effect_mark_unsupported_strategy") !=
        std::string::npos);
}
