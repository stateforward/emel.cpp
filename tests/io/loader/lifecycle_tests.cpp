#include <cstdint>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <doctest/doctest.h>

#include "emel/io/loader/actions.hpp"
#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/loader/guards.hpp"
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

TEST_CASE("io loader action and guard contract covers publication effects") {
  owner_state owner{};
  emel::io::loader::action::context action_ctx{};
  emel::io::loader::detail::runtime_status status{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 4,
      .file_index = 9u,
      .file_offset = 1024u,
      .byte_size = 256u,
      .target = fake_target(0xE000u),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::external_buffer,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};
  emel::io::loader::detail::load_tensor_runtime runtime{request, status};

  CHECK(emel::io::loader::guard::tensor_span_valid{}(runtime, action_ctx));
  CHECK_FALSE(
      emel::io::loader::guard::tensor_span_invalid{}(runtime, action_ctx));
  CHECK_FALSE(emel::io::loader::guard::strategy_none{}(runtime));
  CHECK_FALSE(emel::io::loader::guard::strategy_mapped_file{}(runtime));
  CHECK_FALSE(emel::io::loader::guard::strategy_staged_read{}(runtime));
  CHECK(emel::io::loader::guard::strategy_external_buffer{}(runtime));
  CHECK(emel::io::loader::guard::done_callback_absent{}(runtime));
  CHECK(emel::io::loader::guard::error_callback_absent{}(runtime));

  request.on_done = {&owner, on_load_done};
  request.on_error = {&owner, on_load_error};
  CHECK(emel::io::loader::guard::done_callback_present{}(runtime));
  CHECK(emel::io::loader::guard::error_callback_present{}(runtime));

  emel::io::loader::action::effect_begin_load_tensor(runtime, action_ctx);
  CHECK(status.err == emel::error::cast(emel::io::loader::error::none));
  CHECK_FALSE(status.ok);

  emel::io::loader::action::effect_record_load_tensor_done(runtime, action_ctx);
  CHECK(status.err == emel::error::cast(emel::io::loader::error::none));
  CHECK(status.ok);

  emel::io::loader::action::effect_publish_load_tensor_done(runtime,
                                                            action_ctx);
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.strategy ==
        emel::io::loader::event::strategy_kind::external_buffer);
  CHECK(owner.buffer == fake_target(0xE000u));
  CHECK(owner.buffer_bytes == 256u);

  emel::io::loader::action::effect_mark_invalid_request(runtime, action_ctx);
  CHECK(status.err ==
        emel::error::cast(emel::io::loader::error::invalid_request));
  CHECK_FALSE(status.ok);

  emel::io::loader::action::effect_mark_unsupported_strategy(runtime,
                                                             action_ctx);
  CHECK(status.err ==
        emel::error::cast(emel::io::loader::error::unsupported_strategy));
  CHECK_FALSE(status.ok);

  emel::io::loader::action::effect_publish_load_tensor_error(runtime,
                                                             action_ctx);
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::unsupported_strategy));

  emel::io::loader::action::effect_record_load_tensor_error(runtime,
                                                            action_ctx);
  emel::io::loader::action::effect_on_unexpected(runtime, action_ctx);
  CHECK(status.err ==
        emel::error::cast(emel::io::loader::error::internal_error));
  CHECK_FALSE(status.ok);
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
