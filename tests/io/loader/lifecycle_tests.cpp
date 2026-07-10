#include <cstdint>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/source/any.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/machines.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::io::loader::error::none);
  emel::error::type strategy_err =
      emel::error::cast(emel::io::read::error::none);
  emel::io::loader::event::strategy_kind strategy =
      emel::io::loader::event::strategy_kind::none;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  uint32_t done_count = 0u;
  uint64_t bytes_done = 0u;
  uint32_t failed_index = 0u;
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
  owner->strategy_err = ev.strategy_err;
}

void on_load_batch_done(
    void *object,
    const emel::io::loader::events::load_tensor_batch_done &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->strategy = ev.strategy;
  owner->done_count = ev.done_count;
  owner->bytes_done = ev.bytes_done;
}

void on_load_batch_error(
    void *object,
    const emel::io::loader::events::load_tensor_batch_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
  owner->strategy_err = ev.strategy_err;
  owner->failed_index = ev.failed_index;
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

std::string_view function_source(const std::string &source,
                                 const std::string_view function_name) {
  const size_t name_pos = source.find(function_name);
  REQUIRE(name_pos != std::string::npos);

  const size_t body_begin = source.find('{', name_pos);
  REQUIRE(body_begin != std::string::npos);

  size_t depth = 0u;
  for (size_t cursor = body_begin; cursor < source.size(); ++cursor) {
    if (source[cursor] == '{') {
      depth += 1u;
    } else if (source[cursor] == '}') {
      REQUIRE(depth > 0u);
      depth -= 1u;
      if (depth == 0u) {
        return std::string_view{source.data() + name_pos,
                                cursor + 1u - name_pos};
      }
    }
  }

  FAIL("function body not closed");
  return {};
}

uint32_t occurrence_count(const std::string_view source,
                          const std::string_view needle) {
  uint32_t count = 0u;
  size_t cursor = source.find(needle);
  while (cursor != std::string_view::npos) {
    count += 1u;
    cursor = source.find(needle, cursor + needle.size());
  }
  return count;
}

} // namespace

TEST_CASE("io loader exposes canonical machine aliases") {
  emel::io::loader::sm loader{};
  emel::IoLoader top_level_loader{};

  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
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
      .target_bytes = 64u,
  };
  const std::array strategies{
      emel::io::loader::event::strategy_kind::none,
      emel::io::loader::event::strategy_kind::mapped_file,
      emel::io::loader::event::strategy_kind::read_copy,
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
      .target_bytes = 128u,
  };
  const emel::io::loader::event::strategy_policy unknown_policy{
      static_cast<emel::io::loader::event::strategy_kind>(0xFFu),
  };
  emel::io::loader::event::load_tensor unknown_request{tensor, unknown_policy};
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
      emel::io::loader::event::strategy_kind::read_copy,
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

TEST_CASE("io loader dispatches read/copy requests through the read actor") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 5,
      .file_index = 1u,
      .file_offset = 2u,
      .byte_size = target.size(),
      .file_path = "fixtures.bin",
      .source_buffer = source,
      .source_buffer_bytes = sizeof(source) - 1u,
      .target = target.data(),
      .target_bytes = target.size(),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};
  request.on_done = {&owner, on_load_done};
  request.on_error = {&owner, on_load_error};

  CHECK(loader.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.strategy == emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.buffer == target.data());
  CHECK(owner.buffer_bytes == target.size());
  CHECK(target[0] == 'c');
  CHECK(target[1] == 'd');
  CHECK(target[2] == 'e');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader dispatches staged-read requests through staged actor") {
  emel::io::staged_read::sm staged_actor{};
  emel::io::loader::sm loader{{.io_staged_read = &staged_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdefgh";
  std::array<char, 4> target{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 17,
      .file_index = 1u,
      .file_offset = 2u,
      .byte_size = target.size(),
      .file_path = "fixtures.bin",
      .source_buffer = source,
      .source_buffer_bytes = sizeof(source) - 1u,
      .target = target.data(),
      .target_bytes = target.size(),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::staged_read,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};
  request.on_done = {&owner, on_load_done};
  request.on_error = {&owner, on_load_error};

  CHECK(loader.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.strategy == emel::io::loader::event::strategy_kind::staged_read);
  CHECK(owner.buffer == target.data());
  CHECK(owner.buffer_bytes == target.size());
  CHECK(target[0] == 'c');
  CHECK(target[1] == 'd');
  CHECK(target[2] == 'e');
  CHECK(target[3] == 'f');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader accepts read/copy completion without done callback") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 8,
      .file_index = 1u,
      .file_offset = 1u,
      .byte_size = target.size(),
      .file_path = "fixtures.bin",
      .source_buffer = source,
      .source_buffer_bytes = sizeof(source) - 1u,
      .target = target.data(),
      .target_bytes = target.size(),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};

  CHECK(loader.process_event(request));
  CHECK(target[0] == 'b');
  CHECK(target[1] == 'c');
  CHECK(target[2] == 'd');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader read copy batch routes once through io read") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdefghij";
  std::array<char, 3> first_target{};
  std::array<char, 4> second_target{};
  const std::array<emel::io::loader::event::tensor_load_span, 2> tensors{{
      {
          .tensor_id = 10,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = first_target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = first_target.data(),
          .target_bytes = first_target.size(),
      },
      {
          .tensor_id = 11,
          .file_index = 1u,
          .file_offset = 5u,
          .byte_size = second_target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = second_target.data(),
          .target_bytes = second_target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};
  request.on_done = {&owner, on_load_batch_done};
  request.on_error = {&owner, on_load_batch_error};

  CHECK(loader.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.strategy == emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.done_count == tensors.size());
  CHECK(owner.bytes_done == first_target.size() + second_target.size());
  CHECK(first_target[0] == 'b');
  CHECK(first_target[1] == 'c');
  CHECK(first_target[2] == 'd');
  CHECK(second_target[0] == 'f');
  CHECK(second_target[1] == 'g');
  CHECK(second_target[2] == 'h');
  CHECK(second_target[3] == 'i');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader staged-read batch routes once through staged actor") {
  emel::io::staged_read::sm staged_actor{};
  emel::io::loader::sm loader{{.io_staged_read = &staged_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdefghij";
  std::array<char, 3> first_target{};
  std::array<char, 4> second_target{};
  const std::array<emel::io::loader::event::tensor_load_span, 2> tensors{{
      {
          .tensor_id = 18,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = first_target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = first_target.data(),
          .target_bytes = first_target.size(),
      },
      {
          .tensor_id = 19,
          .file_index = 1u,
          .file_offset = 5u,
          .byte_size = second_target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = second_target.data(),
          .target_bytes = second_target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::staged_read,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};
  request.on_done = {&owner, on_load_batch_done};
  request.on_error = {&owner, on_load_batch_error};

  CHECK(loader.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.strategy == emel::io::loader::event::strategy_kind::staged_read);
  CHECK(owner.done_count == tensors.size());
  CHECK(owner.bytes_done == first_target.size() + second_target.size());
  CHECK(first_target[0] == 'b');
  CHECK(first_target[1] == 'c');
  CHECK(first_target[2] == 'd');
  CHECK(second_target[0] == 'f');
  CHECK(second_target[1] == 'g');
  CHECK(second_target[2] == 'h');
  CHECK(second_target[3] == 'i');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader read copy batch fails closed without read actor") {
  emel::io::loader::sm loader{};
  owner_state owner{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const std::array<emel::io::loader::event::tensor_load_span, 1> tensors{{
      {
          .tensor_id = 12,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = target.data(),
          .target_bytes = target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};
  request.on_done = {&owner, on_load_batch_done};
  request.on_error = {&owner, on_load_batch_error};

  CHECK_FALSE(loader.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::unsupported_strategy));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader read copy batch rejects invalid spans") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const std::array<emel::io::loader::event::tensor_load_span, 1> tensors{{
      {
          .tensor_id = 13,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = 0u,
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = target.data(),
          .target_bytes = target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};
  request.on_done = {&owner, on_load_batch_done};
  request.on_error = {&owner, on_load_batch_error};

  CHECK_FALSE(loader.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::loader::error::invalid_request));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader read copy batch accepts success without done callback") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const std::array<emel::io::loader::event::tensor_load_span, 1> tensors{{
      {
          .tensor_id = 14,
          .file_index = 1u,
          .file_offset = 2u,
          .byte_size = target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = target.data(),
          .target_bytes = target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};

  CHECK(loader.process_event(request));
  CHECK(target[0] == 'c');
  CHECK(target[1] == 'd');
  CHECK(target[2] == 'e');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader read copy batch reports read failures") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const std::array<emel::io::loader::event::tensor_load_span, 1> tensors{{
      {
          .tensor_id = 15,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .source_error =
              emel::error::cast(emel::io::read::error::file_read_failed),
          .target = target.data(),
          .target_bytes = target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};
  request.on_done = {&owner, on_load_batch_done};
  request.on_error = {&owner, on_load_batch_error};

  CHECK_FALSE(loader.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::loader::error::unavailable));
  CHECK(owner.strategy_err ==
        emel::error::cast(emel::io::read::error::file_read_failed));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE(
    "io loader read copy batch rejects missing actor without error callback") {
  emel::io::loader::sm loader{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const std::array<emel::io::loader::event::tensor_load_span, 1> tensors{{
      {
          .tensor_id = 16,
          .file_index = 1u,
          .file_offset = 1u,
          .byte_size = target.size(),
          .file_path = "fixtures.bin",
          .source_buffer = source,
          .source_buffer_bytes = sizeof(source) - 1u,
          .target = target.data(),
          .target_bytes = target.size(),
      },
  }};
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor_batch request{tensors, policy};

  CHECK_FALSE(loader.process_event(request));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader rejects unsupported strategy without error callback") {
  emel::io::loader::sm loader{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 9,
      .file_index = 1u,
      .file_offset = 4096u,
      .byte_size = 64u,
      .target = fake_target(0xE000u),
      .target_bytes = 64u,
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::mapped_file,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};

  CHECK_FALSE(loader.process_event(request));
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}

TEST_CASE("io loader reports read/copy failures through strategy error") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  owner_state owner{};
  constexpr char source[] = "abcdef";
  std::array<char, 3> target{};
  const emel::io::loader::event::tensor_load_span tensor{
      .tensor_id = 6,
      .file_index = 1u,
      .file_offset = 2u,
      .byte_size = target.size(),
      .file_path = "fixtures.bin",
      .source_buffer = source,
      .source_buffer_bytes = sizeof(source) - 1u,
      .source_error =
          emel::error::cast(emel::io::read::error::file_read_failed),
      .target = target.data(),
      .target_bytes = target.size(),
  };
  const emel::io::loader::event::strategy_policy policy{
      emel::io::loader::event::strategy_kind::read_copy,
  };
  emel::io::loader::event::load_tensor request{tensor, policy};
  request.on_done = {&owner, on_load_done};
  request.on_error = {&owner, on_load_error};

  CHECK_FALSE(loader.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::loader::error::unavailable));
  CHECK(owner.strategy_err ==
        emel::error::cast(emel::io::read::error::file_read_failed));
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
  CHECK(actions_source.find("requires { ev.status.err; }") !=
        std::string::npos);
  CHECK(actions_source.find(
            "ev.status.err = emel::error::cast(error::internal_error)") !=
        std::string::npos);
  CHECK(sm_source.find("strategy_mapped_file") != std::string::npos);
  CHECK(sm_source.find("strategy_read_copy") != std::string::npos);
  CHECK(sm_source.find("strategy_read_copy_with_actor") != std::string::npos);
  CHECK(sm_source.find("strategy_external_buffer") != std::string::npos);
  CHECK(sm_source.find("effect_dispatch_read_tensor") != std::string::npos);
  CHECK(sm_source.find("effect_mark_unsupported_strategy") !=
        std::string::npos);
}

TEST_CASE("io loader batch route dispatches one selected read actor event") {
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "loader" / "actions.hpp");
  const std::string_view dispatch_source =
      function_source(actions_source, "effect_dispatch_read_tensor_batch");

  CHECK(actions_source.find("effect_dispatch_read_tensor_batch") !=
        std::string::npos);
  CHECK(occurrence_count(dispatch_source, "process_event(read)") == 1u);
  CHECK(dispatch_source.find("for (") == std::string_view::npos);
}

TEST_CASE("public io source loads bytes through 64-bit file size path") {
  const auto path =
      std::filesystem::temp_directory_path() / "emel_io_source_bytes.bin";
  std::filesystem::remove(path);
  {
    std::ofstream output{path, std::ios::binary};
    REQUIRE(output.good());
    output << "source-bytes";
  }

  std::vector<uint8_t> bytes;
  CHECK(emel::io::source::load_file_bytes(path.string(), bytes) ==
        emel::error::cast(emel::io::read::error::none));
  CHECK(std::string{bytes.begin(), bytes.end()} == "source-bytes");
  std::filesystem::remove(path);

  const std::string source = read_text_file(repo_root() / "src" / "emel" /
                                            "io" / "source" / "any.hpp");
  CHECK(source.find("std::filesystem::file_size") != std::string::npos);
  CHECK(source.find("std::ftell") == std::string::npos);
}

TEST_CASE("public io source validates paths before opening") {
  std::vector<uint8_t> bytes{1u, 2u, 3u};

  CHECK(emel::io::source::load_file_bytes({}, bytes) ==
        emel::error::cast(emel::io::read::error::invalid_request));
  CHECK(bytes.empty());

  const std::string overlong_path(
      static_cast<size_t>(emel::io::read::k_max_file_path_bytes + 1u), 'x');
  CHECK(emel::io::source::load_file_bytes(overlong_path, bytes) ==
        emel::error::cast(emel::io::read::error::invalid_request));

  const std::string embedded_nul_path{"source\0bytes", 12u};
  CHECK(emel::io::source::load_file_bytes(embedded_nul_path, bytes) ==
        emel::error::cast(emel::io::read::error::invalid_request));
}

TEST_CASE("public io source reports open failures") {
  const auto path =
      std::filesystem::temp_directory_path() / "emel_io_source_missing.bin";
  std::filesystem::remove(path);

  std::vector<uint8_t> bytes;
  CHECK(emel::io::source::load_file_bytes(path.string(), bytes) ==
        emel::error::cast(emel::io::read::error::file_open_failed));
  CHECK(bytes.empty());
}

TEST_CASE("public io source reports file size failures") {
  const auto path =
      std::filesystem::temp_directory_path() / "emel_io_source_directory";
  std::filesystem::remove_all(path);
  std::filesystem::create_directory(path);

  std::vector<uint8_t> bytes;
  const emel::error::type result =
      emel::io::source::load_file_bytes(path.string(), bytes);
  CHECK((result == emel::error::cast(emel::io::read::error::file_seek_failed) ||
         result == emel::error::cast(emel::io::read::error::file_open_failed)));
  std::filesystem::remove_all(path);
}

TEST_CASE("public io source rejects oversized files before allocation") {
  const auto path =
      std::filesystem::temp_directory_path() / "emel_io_source_oversized.bin";
  std::filesystem::remove(path);
  {
    std::ofstream output{path, std::ios::binary};
    REQUIRE(output.good());
  }

  std::error_code resize_error;
  std::filesystem::resize_file(path, emel::io::read::k_max_read_bytes + 1u,
                               resize_error);
  REQUIRE_FALSE(resize_error);

  std::vector<uint8_t> bytes;
  CHECK(emel::io::source::load_file_bytes(path.string(), bytes) ==
        emel::error::cast(emel::io::read::error::unsupported_resource));
  CHECK(bytes.empty());
  std::filesystem::remove(path);
}

TEST_CASE(
    "io loader exposes staged-read strategy naming via public contracts") {
  const auto root = repo_root();
  const std::array runtime_sources{
      "src/emel/io/loader/events.hpp",
      "src/emel/io/loader/guards.hpp",
      "src/emel/io/loader/sm.hpp",
      "tools/bench/model_load_strategy.hpp",
  };

  const std::string events_source =
      read_text_file(root / "src" / "emel" / "io" / "loader" / "events.hpp");
  const std::string guards_source =
      read_text_file(root / "src" / "emel" / "io" / "loader" / "guards.hpp");
  const std::string sm_source =
      read_text_file(root / "src" / "emel" / "io" / "loader" / "sm.hpp");
  const std::string helper_source =
      read_text_file(root / "tools" / "bench" / "model_load_strategy.hpp");

  CHECK(events_source.find("read_copy = 2u") != std::string::npos);
  CHECK(events_source.find("staged_read = 4u") != std::string::npos);
  CHECK(guards_source.find("strategy_read_copy_with_actor") !=
        std::string::npos);
  CHECK(guards_source.find("strategy_staged_read_with_actor") !=
        std::string::npos);
  CHECK(sm_source.find("effect_dispatch_staged_read_tensor") !=
        std::string::npos);
  CHECK(sm_source.find("effect_dispatch_staged_read_tensor_batch") !=
        std::string::npos);
  CHECK(helper_source.find("\"read_copy\"") != std::string::npos);
  CHECK(helper_source.find("\"staged_read\"") != std::string::npos);
  CHECK(events_source.find("staged_chunk_bytes") != std::string::npos);

  for (const auto *source_path : runtime_sources) {
    CAPTURE(source_path);
    const std::string source = read_text_file(root / source_path);
    CHECK(source.find("emel/io/staged_read/detail.hpp") == std::string::npos);
    CHECK(source.find("emel/io/staged_read/actions.hpp") == std::string::npos);
    CHECK(source.find("emel/io/staged_read/guards.hpp") == std::string::npos);
  }
}

TEST_CASE("io loader staged-read dispatch uses bounded chunk policy") {
  const auto root = repo_root();
  const std::string actions_source =
      read_text_file(root / "src" / "emel" / "io" / "loader" / "actions.hpp");
  const std::string detail_source =
      read_text_file(root / "src" / "emel" / "io" / "loader" / "detail.hpp");
  const std::string staged_events_source = read_text_file(
      root / "src" / "emel" / "io" / "staged_read" / "events.hpp");

  CHECK(actions_source.find(".stage_chunk_bytes = tensor.byte_size") ==
        std::string::npos);
  CHECK(actions_source.find("compute_staged_chunk_bytes") != std::string::npos);
  CHECK(detail_source.find("compute_staged_chunk_bytes") != std::string::npos);
  CHECK(staged_events_source.find("stage_chunk_bytes") != std::string::npos);
}
