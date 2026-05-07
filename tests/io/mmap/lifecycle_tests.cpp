#include <cstdint>
#include <cstring>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "emel/io/mmap/actions.hpp"
#include "emel/io/mmap/detail.hpp"
#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/guards.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/machines.hpp"

namespace {

struct map_owner_state {
  bool done = false;
  bool error = false;
  uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  emel::error::type err = emel::error::cast(emel::io::mmap::error::none);
};

struct release_owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::io::mmap::error::none);
};

void on_map_done(void *object,
                 const emel::io::mmap::events::map_tensor_done &ev) noexcept {
  auto *owner = static_cast<map_owner_state *>(object);
  owner->done = true;
  owner->handle = ev.handle;
  owner->buffer = ev.buffer;
  owner->buffer_bytes = ev.buffer_bytes;
}

void on_map_error(void *object,
                  const emel::io::mmap::events::map_tensor_error &ev) noexcept {
  auto *owner = static_cast<map_owner_state *>(object);
  owner->error = true;
  owner->err = ev.err;
}

void on_release_done(
    void *object,
    const emel::io::mmap::events::release_mapping_done &) noexcept {
  auto *owner = static_cast<release_owner_state *>(object);
  owner->done = true;
}

void on_release_error(
    void *object,
    const emel::io::mmap::events::release_mapping_error &ev) noexcept {
  auto *owner = static_cast<release_owner_state *>(object);
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

std::filesystem::path make_temp_file(std::string_view tag,
                                     const std::vector<uint8_t> &payload) {
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string{"emel_io_mmap_"} + std::string{tag} + ".bin");
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  REQUIRE(out.good());
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char *>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
  }
  out.close();
  return path;
}

std::vector<uint8_t> make_payload(uint64_t bytes, uint8_t seed) {
  std::vector<uint8_t> data(static_cast<size_t>(bytes));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>((seed + (i & 0xFFu)) & 0xFFu);
  }
  return data;
}

struct unrelated_event {};

} // namespace

TEST_CASE("io mmap exposes canonical machine aliases at component boundary") {
  emel::io::mmap::sm strategy{};
  emel::IoMmap top_level_mmap{};

  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
  CHECK(
      top_level_mmap.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap reports state_ready via visit_current_states after a full "
          "map-then-release dispatch") {
  emel::io::mmap::sm strategy{};
  map_owner_state map_owner{};
  release_owner_state release_owner{};

  const auto payload = make_payload(4096u, 0x33u);
  const auto path = make_temp_file("visit_current_states", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 600,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&map_owner, on_map_done};
  map_request.on_error = {&map_owner, on_map_error};
  REQUIRE(strategy.process_event(map_request));

  emel::io::mmap::event::release_mapping release_request{600, map_owner.handle};
  release_request.on_done = {&release_owner, on_release_done};
  release_request.on_error = {&release_owner, on_release_error};
  REQUIRE(strategy.process_event(release_request));
  REQUIRE(release_owner.done);

  // RTC: dispatch returns to state_ready with no residual decision states.
  std::size_t visited_states = 0;
  bool saw_ready = false;
  strategy.visit_current_states([&](auto state) noexcept {
    ++visited_states;
    using state_t = typename decltype(state)::type;
    if constexpr (std::is_same_v<state_t, emel::io::mmap::state_ready>) {
      saw_ready = true;
    }
  });
  CHECK(visited_states == 1);
  CHECK(saw_ready);
  std::filesystem::remove(path);
}

TEST_CASE("io mmap validation rejection does not consume a slot") {
  emel::io::mmap::sm strategy{};

  // Burn through every reject path: each must leave the slot pool untouched.
  const std::string ok_path = "/tmp/emel_io_mmap_does_not_matter.bin";
  std::vector<emel::io::mmap::event::map_tensor_request> rejects;
  rejects.push_back({.tensor_id = 1,
                     .file_index = 0u,
                     .file_offset = 0u,
                     .byte_size = 0u,
                     .file_path = ok_path}); // invalid_request
  rejects.push_back({.tensor_id = 2,
                     .file_index = 0u,
                     .file_offset = 0u,
                     .byte_size = 1024u,
                     .file_path = {}}); // empty file_path
  rejects.push_back(
      {.tensor_id = 3,
       .file_index =
           static_cast<uint16_t>(emel::io::mmap::k_max_file_index + 1u),
       .file_offset = 0u,
       .byte_size = 1024u,
       .file_path = ok_path}); // unsupported_resource (file_index)
  rejects.push_back({.tensor_id = 4,
                     .file_index = 0u,
                     .file_offset = 17u,
                     .byte_size = 1024u,
                     .file_path = ok_path}); // unsupported_resource (offset)

  for (const auto &r : rejects) {
    map_owner_state owner{};
    emel::io::mmap::event::map_tensor map_request{r};
    map_request.on_done = {&owner, on_map_done};
    map_request.on_error = {&owner, on_map_error};
    CHECK_FALSE(strategy.process_event(map_request));
    CHECK(owner.error);
  }

  // The slot pool must still be empty: a fresh successful map must use the
  // same first-free handle every time, and a full k_max_mappings batch must
  // succeed.
  const auto payload = make_payload(4096u, 0x44u);
  const auto file_path = make_temp_file("rejects_no_slot", payload);
  const std::string file_path_str = file_path.string();
  const emel::io::mmap::event::map_tensor_request good{
      .tensor_id = 9001,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = file_path_str,
  };

  std::vector<uint32_t> taken_handles;
  taken_handles.reserve(emel::io::mmap::k_max_mappings);
  for (uint32_t i = 0; i < emel::io::mmap::k_max_mappings; ++i) {
    map_owner_state owner{};
    emel::io::mmap::event::map_tensor map_request{good};
    map_request.on_done = {&owner, on_map_done};
    map_request.on_error = {&owner, on_map_error};
    REQUIRE(strategy.process_event(map_request));
    taken_handles.push_back(owner.handle);
  }

  for (uint32_t h : taken_handles) {
    emel::io::mmap::event::release_mapping cleanup{9001, h};
    CHECK(strategy.process_event(cleanup));
  }
  std::filesystem::remove(file_path);
}

TEST_CASE("io mmap rejects invalid request spans before any mapping attempt") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_does_not_matter.bin";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 9,
      .file_index = 1u,
      .file_offset = 4096u,
      .byte_size = 0u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::mmap::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap rejects empty file_path as invalid_request") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 7,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 1024u,
      .file_path = {},
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::mmap::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap rejects map spans beyond file size before mapping") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const auto payload = make_payload(4096u, 0x5Au);
  const auto path = make_temp_file("span_beyond_eof", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 9,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 8192u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
  std::filesystem::remove(path);
}

TEST_CASE("io mmap rejects embedded NUL file_path as invalid_request") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  std::string path_with_nul = "/tmp/emel_io_mmap";
  path_with_nul.push_back('\0');
  path_with_nul += "hidden.bin";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 8,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 1024u,
      .file_path = path_with_nul,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::mmap::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap rejects out-of-range file_index as unsupported resource") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_does_not_matter.bin";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 21,
      .file_index =
          static_cast<uint16_t>(emel::io::mmap::k_max_file_index + 1u),
      .file_offset = 0u,
      .byte_size = 1024u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap rejects unaligned file_offset as unsupported resource") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_does_not_matter.bin";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 22,
      .file_index = 0u,
      .file_offset = 17u,
      .byte_size = 1024u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap offset guard uses actor-local required alignment") {
  emel::io::mmap::action::context ctx{};
  ctx.required_offset_alignment =
      emel::io::mmap::k_required_offset_alignment * 2u;

  emel::io::mmap::detail::map_attempt_status status{};
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 220,
      .file_index = 0u,
      .file_offset = emel::io::mmap::k_required_offset_alignment,
      .byte_size = 1024u,
      .file_path = "/tmp/emel_io_mmap_alignment_guard.bin",
  };
  const emel::io::mmap::event::map_tensor map_request{request};
  const emel::io::mmap::detail::map_tensor_runtime runtime{map_request, status};

  CHECK_FALSE(emel::io::mmap::guard::offset_aligned{}(runtime, ctx));
  CHECK(emel::io::mmap::guard::offset_unaligned{}(runtime, ctx));
}

TEST_CASE("io mmap rejects byte_size above maximum as unsupported resource") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_does_not_matter.bin";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 23,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = emel::io::mmap::k_max_mapping_bytes + 1u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap rejects layouts that overflow the address space") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_does_not_matter.bin";
  constexpr uint64_t addr_max = static_cast<uint64_t>(-1);
  constexpr uint64_t big_size = emel::io::mmap::k_max_mapping_bytes;
  constexpr uint64_t big_offset =
      ((addr_max - big_size) + emel::io::mmap::k_required_offset_alignment) &
      ~(emel::io::mmap::k_required_offset_alignment - 1u);
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 24,
      .file_index = 0u,
      .file_offset = big_offset,
      .byte_size = big_size,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap surfaces file_open_failed when the path does not exist") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_definitely_missing_xyzzy.bin";
  std::filesystem::remove(path);
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 31,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 1024u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::file_open_failed));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap returns a deterministic mapped descriptor on success") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const auto payload = make_payload(4096u, 0x42u);
  const auto path = make_temp_file("success", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 1001,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  const bool ok = strategy.process_event(map_request);
  CHECK(ok);
  CHECK_FALSE(owner.error);
  CHECK(owner.done);
  CHECK(owner.handle != emel::io::mmap::k_invalid_mapping_handle);
  REQUIRE(owner.buffer != nullptr);
  CHECK(owner.buffer_bytes == 4096u);
  CHECK(static_cast<const uint8_t *>(owner.buffer)[0] == payload[0]);
  CHECK(static_cast<const uint8_t *>(owner.buffer)[4095] == payload[4095]);
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  emel::io::mmap::event::release_mapping release_request{1001, owner.handle};
  CHECK(strategy.process_event(release_request));
  std::filesystem::remove(path);
}

TEST_CASE("io mmap rejects map_tensor without done callback") {
  const auto payload = make_payload(4096u, 0x5Au);
  const auto path = make_temp_file("missing_done_callback", payload);
  const std::string path_str = path.string();

  {
    emel::io::mmap::sm strategy{};
    map_owner_state owner{};
    const emel::io::mmap::event::map_tensor_request request{
        .tensor_id = 1002,
        .file_index = 0u,
        .file_offset = 0u,
        .byte_size = 4096u,
        .file_path = path_str,
    };
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_error = {&owner, on_map_error};

    CHECK_FALSE(strategy.process_event(map_request));
    CHECK_FALSE(owner.done);
    CHECK(owner.error);
    CHECK(owner.err ==
          emel::error::cast(emel::io::mmap::error::invalid_request));
    CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
  }

  std::filesystem::remove(path);
}

TEST_CASE("io mmap actor destruction releases active mappings") {
  const auto payload = make_payload(4096u, 0x24u);
  const auto path = make_temp_file("destructor_releases", payload);
  const std::string path_str = path.string();

  {
    emel::io::mmap::sm strategy{};
    map_owner_state owner{};
    const emel::io::mmap::event::map_tensor_request request{
        .tensor_id = 1003,
        .file_index = 0u,
        .file_offset = 0u,
        .byte_size = 4096u,
        .file_path = path_str,
    };
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_done = {&owner, on_map_done};
    map_request.on_error = {&owner, on_map_error};

    REQUIRE(strategy.process_event(map_request));
    CHECK(owner.handle != emel::io::mmap::k_invalid_mapping_handle);
    CHECK(owner.buffer != nullptr);
  }

  CHECK(std::filesystem::remove(path));
}

TEST_CASE(
    "io mmap copies non-terminated file_path views before platform open") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  const auto payload = make_payload(4096u, 0x66u);
  const auto path = make_temp_file("sliced_path", payload);
  const std::string path_str = path.string();
  const std::string path_with_suffix = path_str + "_suffix";
  const std::string_view sliced_path{path_with_suffix.data(), path_str.size()};
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 1002,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = sliced_path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};

  CHECK(strategy.process_event(map_request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  REQUIRE(owner.buffer != nullptr);
  CHECK(static_cast<const uint8_t *>(owner.buffer)[0] == payload[0]);

  emel::io::mmap::event::release_mapping release_request{1002, owner.handle};
  CHECK(strategy.process_event(release_request));
  std::filesystem::remove(path);
}

TEST_CASE("io mmap release happy path returns slot to the free pool") {
  emel::io::mmap::sm strategy{};
  map_owner_state map_owner{};
  release_owner_state release_owner{};

  const auto payload = make_payload(4096u, 0x11u);
  const auto path = make_temp_file("release_happy", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 200,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&map_owner, on_map_done};
  map_request.on_error = {&map_owner, on_map_error};
  CHECK(strategy.process_event(map_request));
  REQUIRE(map_owner.handle != emel::io::mmap::k_invalid_mapping_handle);

  emel::io::mmap::event::release_mapping release_request{200, map_owner.handle};
  release_request.on_done = {&release_owner, on_release_done};
  release_request.on_error = {&release_owner, on_release_error};
  CHECK(strategy.process_event(release_request));
  CHECK(release_owner.done);
  CHECK_FALSE(release_owner.error);
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  // The released slot must be reusable; map again and observe LIFO reuse.
  map_owner_state second_owner{};
  emel::io::mmap::event::map_tensor second_map{request};
  second_map.on_done = {&second_owner, on_map_done};
  second_map.on_error = {&second_owner, on_map_error};
  CHECK(strategy.process_event(second_map));
  CHECK(second_owner.handle == map_owner.handle);

  emel::io::mmap::event::release_mapping cleanup{200, second_owner.handle};
  CHECK(strategy.process_event(cleanup));
  std::filesystem::remove(path);
}

TEST_CASE("io mmap release rejects handles owned by another tensor") {
  emel::io::mmap::sm strategy{};
  map_owner_state map_owner{};
  release_owner_state wrong_owner{};

  const auto payload = make_payload(4096u, 0x12u);
  const auto path = make_temp_file("release_wrong_tensor", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 700,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&map_owner, on_map_done};
  map_request.on_error = {&map_owner, on_map_error};
  REQUIRE(strategy.process_event(map_request));

  emel::io::mmap::event::release_mapping wrong_release{701, map_owner.handle};
  wrong_release.on_error = {&wrong_owner, on_release_error};
  CHECK_FALSE(strategy.process_event(wrong_release));
  CHECK(wrong_owner.error);
  CHECK(wrong_owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_request));

  emel::io::mmap::event::release_mapping cleanup{700, map_owner.handle};
  CHECK(strategy.process_event(cleanup));
  std::filesystem::remove(path);
}

TEST_CASE("io mmap pre-map failure actions release reserved slot") {
  emel::io::mmap::action::context ctx{};
  emel::io::mmap::detail::map_attempt_status status{};
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 77,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = "/tmp/emel_io_mmap_direct_failure.bin",
  };
  const emel::io::mmap::event::map_tensor map_request{request};
  const emel::io::mmap::detail::map_tensor_runtime runtime{map_request, status};

  status.reserved_slot = ctx.free_stack[ctx.free_count - 1u];
  status.os_resource = -1;
  ctx.free_count -= 1u;
  ctx.slots[status.reserved_slot].in_use = true;
  emel::io::mmap::action::
      effect_close_open_resource_and_release_slot_on_mapping_failure(runtime,
                                                                     ctx);

  CHECK(status.err == emel::error::cast(emel::io::mmap::error::mapping_failed));
  CHECK_FALSE(ctx.slots[status.reserved_slot].in_use);
  CHECK(ctx.free_count == emel::io::mmap::k_max_mappings);

  status.reserved_slot = ctx.free_stack[ctx.free_count - 1u];
  status.os_resource = -1;
  ctx.free_count -= 1u;
  ctx.slots[status.reserved_slot].in_use = true;
  emel::io::mmap::action::
      effect_close_open_resource_and_release_slot_on_file_span_failure(runtime,
                                                                       ctx);

  CHECK(status.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_resource));
  CHECK_FALSE(ctx.slots[status.reserved_slot].in_use);
  CHECK(ctx.free_count == emel::io::mmap::k_max_mappings);
}

TEST_CASE("io mmap file-size and failure guards classify raw status") {
  emel::io::mmap::action::context ctx{};
  emel::io::mmap::detail::map_attempt_status map_status{};
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 78,
      .file_index = 0u,
      .file_offset = 4096u,
      .byte_size = 4096u,
      .file_path = "/tmp/emel_io_mmap_guard_failure.bin",
  };
  const emel::io::mmap::event::map_tensor map_request{request};
  const emel::io::mmap::detail::map_tensor_runtime map_runtime{map_request,
                                                               map_status};

  map_status.os_resource = -1;
  emel::io::mmap::action::effect_measure_open_file_size(map_runtime, ctx);
  CHECK_FALSE(map_status.file_size_ok);
  CHECK(emel::io::mmap::guard::file_span_exceeds_file{}(map_runtime, ctx));

  map_status.file_size_ok = true;
  map_status.file_size_bytes = 8192u;
  CHECK(emel::io::mmap::guard::file_span_within_file{}(map_runtime, ctx));

  map_status.mapping_ok = false;
  CHECK(emel::io::mmap::guard::mapping_failed{}(map_runtime, ctx));
  CHECK_FALSE(
      emel::io::mmap::guard::platform_mmap_unsupported{}(map_runtime, ctx));

  emel::io::mmap::detail::release_attempt_status release_status{};
  const emel::io::mmap::event::release_mapping release_request{78, 0u};
  const emel::io::mmap::detail::release_mapping_runtime release_runtime{
      release_request, release_status};
  CHECK(emel::io::mmap::guard::unmap_failed{}(release_runtime, ctx));

  emel::io::mmap::action::effect_mark_unsupported_platform(map_runtime, ctx);
  CHECK(map_status.err ==
        emel::error::cast(emel::io::mmap::error::unsupported_platform));
}

TEST_CASE("io mmap release rejects out-of-range handle") {
  emel::io::mmap::sm strategy{};
  release_owner_state owner{};
  emel::io::mmap::event::release_mapping release_request{
      1, emel::io::mmap::k_max_mappings};
  release_request.on_error = {&owner, on_release_error};

  CHECK_FALSE(strategy.process_event(release_request));
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::io::mmap::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap release rejects double release on the same handle") {
  emel::io::mmap::sm strategy{};
  map_owner_state map_owner{};

  const auto payload = make_payload(4096u, 0x22u);
  const auto path = make_temp_file("release_double", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 300,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&map_owner, on_map_done};
  map_request.on_error = {&map_owner, on_map_error};
  REQUIRE(strategy.process_event(map_request));

  emel::io::mmap::event::release_mapping first_release{300, map_owner.handle};
  CHECK(strategy.process_event(first_release));

  release_owner_state second_owner{};
  emel::io::mmap::event::release_mapping second_release{300, map_owner.handle};
  second_release.on_error = {&second_owner, on_release_error};
  CHECK_FALSE(strategy.process_event(second_release));
  CHECK(second_owner.error);
  CHECK(second_owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  std::filesystem::remove(path);
}

TEST_CASE("io mmap fails closed without an error callback") {
  emel::io::mmap::sm strategy{};
  const std::string path = "/tmp/emel_io_mmap_no_callback.bin";
  const emel::io::mmap::event::map_tensor_request invalid{};
  emel::io::mmap::event::map_tensor invalid_request{invalid};

  CHECK_FALSE(strategy.process_event(invalid_request));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  emel::io::mmap::event::release_mapping invalid_release{
      1, emel::io::mmap::k_max_mappings};
  CHECK_FALSE(strategy.process_event(invalid_release));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap surfaces resource_exhausted when slot pool is full") {
  emel::io::mmap::sm strategy{};
  const auto payload = make_payload(4096u, 0x55u);
  const auto path = make_temp_file("resource_exhausted", payload);
  const std::string path_str = path.string();
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 50,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path_str,
  };

  std::vector<uint32_t> taken_handles;
  taken_handles.reserve(emel::io::mmap::k_max_mappings);
  for (uint32_t i = 0; i < emel::io::mmap::k_max_mappings; ++i) {
    map_owner_state owner{};
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_done = {&owner, on_map_done};
    map_request.on_error = {&owner, on_map_error};
    REQUIRE(strategy.process_event(map_request));
    taken_handles.push_back(owner.handle);
  }

  map_owner_state exhausted_owner{};
  emel::io::mmap::event::map_tensor exhausted_request{request};
  exhausted_request.on_done = {&exhausted_owner, on_map_done};
  exhausted_request.on_error = {&exhausted_owner, on_map_error};
  CHECK_FALSE(strategy.process_event(exhausted_request));
  CHECK(exhausted_owner.error);
  CHECK(exhausted_owner.err ==
        emel::error::cast(emel::io::mmap::error::resource_exhausted));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  for (uint32_t h : taken_handles) {
    emel::io::mmap::event::release_mapping cleanup{50, h};
    CHECK(strategy.process_event(cleanup));
  }
  std::filesystem::remove(path);
}

TEST_CASE("io mmap surfaces mapping_failed when mmap call fails") {
  emel::io::mmap::sm strategy{};
  map_owner_state owner{};
  // POSIX permits open() on a directory but mmap() on a directory fd
  // returns EACCES/ENODEV. Platforms may reject it during open, file-size
  // measurement, or mmap; each path must recover to ready and release any slot
  // or fd it acquired.
  const std::string path = "/";
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 500,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 4096u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};
  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.error);
  CHECK((
      owner.err == emel::error::cast(emel::io::mmap::error::mapping_failed) ||
      owner.err == emel::error::cast(emel::io::mmap::error::file_open_failed) ||
      owner.err ==
          emel::error::cast(emel::io::mmap::error::unsupported_resource)));
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap handles unexpected events deterministically") {
  emel::io::mmap::sm strategy{};
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  strategy.process_event(unrelated_event{});
  CHECK(strategy.is(stateforward::sml::state<emel::io::mmap::state_ready>));

  map_owner_state owner{};
  const std::string path = "/tmp/emel_io_mmap_unexpected.bin";
  std::filesystem::remove(path);
  const emel::io::mmap::event::map_tensor_request request{
      .tensor_id = 13,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = 256u,
      .file_path = path,
  };
  emel::io::mmap::event::map_tensor map_request{request};
  map_request.on_done = {&owner, on_map_done};
  map_request.on_error = {&owner, on_map_error};
  CHECK_FALSE(strategy.process_event(map_request));
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::file_open_failed));
}

TEST_CASE("io mmap boundary keeps platform calls inside actions.cpp") {
  const std::string actions_hpp_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "actions.hpp");
  const std::string detail_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "detail.hpp");
  const std::string sm_source =
      read_text_file(repo_root() / "src" / "emel" / "io" / "mmap" / "sm.hpp");
  const std::string guards_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "guards.hpp");
  const std::string events_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "events.hpp");
  const std::string context_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "context.hpp");
  const std::string actions_cpp_source = read_text_file(
      repo_root() / "src" / "emel" / "io" / "mmap" / "actions.cpp");

  // Headers must not include or invoke platform mapping calls.
  for (const std::string *src :
       {&actions_hpp_source, &detail_source, &sm_source, &guards_source,
        &events_source, &context_source}) {
    CHECK(src->find("::mmap(") == std::string::npos);
    CHECK(src->find("munmap(") == std::string::npos);
    CHECK(src->find("MapViewOfFile") == std::string::npos);
    CHECK(src->find("CreateFileMapping") == std::string::npos);
    CHECK(src->find("UnmapViewOfFile") == std::string::npos);
    CHECK(src->find("<sys/mman.h>") == std::string::npos);
    CHECK(src->find("<windows.h>") == std::string::npos);
    CHECK(src->find("<fcntl.h>") == std::string::npos);
  }

  // actions.cpp is the single owner of platform mapping calls.
  CHECK((actions_cpp_source.find("::mmap(") != std::string::npos ||
         actions_cpp_source.find("MapViewOfFile") != std::string::npos));
  CHECK((actions_cpp_source.find("munmap(") != std::string::npos ||
         actions_cpp_source.find("UnmapViewOfFile") != std::string::npos));

  // Validation chain states from Phase 205.
  CHECK(sm_source.find("state_request_decision") != std::string::npos);
  CHECK(sm_source.find("state_file_path_decision") != std::string::npos);
  CHECK(sm_source.find("state_file_decision") != std::string::npos);
  CHECK(sm_source.find("state_offset_decision") != std::string::npos);
  CHECK(sm_source.find("state_length_decision") != std::string::npos);
  CHECK(sm_source.find("state_layout_decision") != std::string::npos);
  CHECK(sm_source.find("state_platform_decision") != std::string::npos);

  // Phase 206 success and lifetime states.
  CHECK(sm_source.find("state_slot_reservation_decision") != std::string::npos);
  CHECK(sm_source.find("state_file_open_decision") != std::string::npos);
  CHECK(sm_source.find("state_mapping_decision") != std::string::npos);
  CHECK(sm_source.find("state_done_callback") != std::string::npos);
  CHECK(sm_source.find("state_release_decision") != std::string::npos);
  CHECK(sm_source.find("state_unmap_decision") != std::string::npos);

  // Phase 206 error decision states.
  CHECK(sm_source.find("state_resource_exhausted_error_decision") !=
        std::string::npos);
  CHECK(sm_source.find("state_file_open_failed_error_decision") !=
        std::string::npos);
  CHECK(sm_source.find("state_mapping_failed_error_decision") !=
        std::string::npos);
  CHECK(sm_source.find("state_release_invalid_handle_error_decision") !=
        std::string::npos);
  CHECK(sm_source.find("state_unmap_failed_error_decision") !=
        std::string::npos);

  // Out-of-scope strategy markers must remain absent.
  CHECK(sm_source.find("strategy_read_copy") == std::string::npos);
  CHECK(sm_source.find("strategy_external_buffer") == std::string::npos);
  CHECK(sm_source.find("strategy_async") == std::string::npos);
  CHECK(sm_source.find("strategy_device") == std::string::npos);
  CHECK(sm_source.find("strategy_copy") == std::string::npos);
}

// PR #83 P1 (PRRT_kwDORRHzJs5_hhbx): when platform_unmap fails, the actor must
// retain ownership of the slot (in_use=true with base/bytes/os_resource intact)
// so the caller can retry release. Pushing the handle back onto free_stack
// would let a later map_tensor reuse the slot while the OS mapping is still
// alive.
TEST_CASE("io mmap unmap failure keeps mapping slot owned for retry") {
  emel::io::mmap::action::context ctx{};
  const uint32_t initial_free_count = ctx.free_count;
  REQUIRE(initial_free_count == emel::io::mmap::k_max_mappings);

  // Reserve slot 0 with a representative live mapping.
  const uint32_t target_slot = 0u;
  ctx.slots[target_slot].in_use = true;
  ctx.slots[target_slot].tensor_id = 4242;
  ctx.slots[target_slot].base = reinterpret_cast<void *>(0xCAFEBABEu);
  ctx.slots[target_slot].mapped_bytes = 4096u;
  ctx.slots[target_slot].os_resource = 17;
  ctx.slots[target_slot].file_offset = 0u;
  ctx.slots[target_slot].requested_bytes = 4096u;
  // Pop the slot from the free stack so free_count reflects an in-use slot.
  for (uint32_t i = 0u; i < ctx.free_count; ++i) {
    if (ctx.free_stack[i] == target_slot) {
      ctx.free_stack[i] = ctx.free_stack[ctx.free_count - 1u];
      ctx.free_count -= 1u;
      break;
    }
  }
  const uint32_t free_count_with_slot_in_use = ctx.free_count;
  REQUIRE(free_count_with_slot_in_use == emel::io::mmap::k_max_mappings - 1u);

  emel::io::mmap::detail::release_attempt_status status{};
  status.target_slot = target_slot;
  status.unmap_base = ctx.slots[target_slot].base;
  status.unmap_bytes = ctx.slots[target_slot].mapped_bytes;
  status.os_resource = ctx.slots[target_slot].os_resource;
  status.unmap_ok = false;
  const emel::io::mmap::event::release_mapping release_request{4242,
                                                               target_slot};
  const emel::io::mmap::detail::release_mapping_runtime runtime{release_request,
                                                                status};

  emel::io::mmap::action::effect_mark_unmap_failed_and_release_slot(runtime,
                                                                    ctx);

  CHECK(status.err == emel::error::cast(emel::io::mmap::error::unmap_failed));
  CHECK_FALSE(status.ok);

  // Slot must remain owned (in_use=true) so the caller can retry release.
  CHECK(ctx.slots[target_slot].in_use);
  CHECK(ctx.slots[target_slot].tensor_id == 4242);
  CHECK(ctx.slots[target_slot].base == reinterpret_cast<void *>(0xCAFEBABEu));
  CHECK(ctx.slots[target_slot].mapped_bytes == 4096u);
  CHECK(ctx.slots[target_slot].os_resource == 17);

  // Free stack must NOT have grown — the slot stays unavailable for reuse.
  CHECK(ctx.free_count == free_count_with_slot_in_use);
  for (uint32_t i = 0u; i < ctx.free_count; ++i) {
    CHECK(ctx.free_stack[i] != target_slot);
  }

  ctx.slots[target_slot].in_use = false;
  ctx.slots[target_slot].base = nullptr;
  ctx.slots[target_slot].mapped_bytes = 0u;
  ctx.slots[target_slot].os_resource = -1;
}

TEST_CASE("io mmap unmap failure clears resources already released") {
  emel::io::mmap::action::context ctx{};
  const uint32_t target_slot = 0u;
  ctx.slots[target_slot].in_use = true;
  ctx.slots[target_slot].tensor_id = 5150;
  ctx.slots[target_slot].base = reinterpret_cast<void *>(0xABCD0000u);
  ctx.slots[target_slot].mapped_bytes = 8192u;
  ctx.slots[target_slot].os_resource = 23;
  ctx.slots[target_slot].file_offset = 4096u;
  ctx.slots[target_slot].requested_bytes = 8192u;
  for (uint32_t i = 0u; i < ctx.free_count; ++i) {
    if (ctx.free_stack[i] == target_slot) {
      ctx.free_stack[i] = ctx.free_stack[ctx.free_count - 1u];
      ctx.free_count -= 1u;
      break;
    }
  }
  const uint32_t free_count_with_slot_in_use = ctx.free_count;

  emel::io::mmap::detail::release_attempt_status status{};
  status.target_slot = target_slot;
  status.unmap_base = ctx.slots[target_slot].base;
  status.unmap_bytes = ctx.slots[target_slot].mapped_bytes;
  status.os_resource = ctx.slots[target_slot].os_resource;
  status.unmap_ok = false;
  status.unmap_base_released = true;
  status.os_resource_released = false;
  const emel::io::mmap::event::release_mapping release_request{5150,
                                                               target_slot};
  const emel::io::mmap::detail::release_mapping_runtime runtime{release_request,
                                                                status};

  emel::io::mmap::action::effect_mark_unmap_failed_and_release_slot(runtime,
                                                                    ctx);

  CHECK(ctx.slots[target_slot].in_use);
  CHECK(ctx.slots[target_slot].base == nullptr);
  CHECK(ctx.slots[target_slot].mapped_bytes == 0u);
  CHECK(ctx.slots[target_slot].os_resource == 23);
  CHECK(ctx.free_count == free_count_with_slot_in_use);

  ctx.slots[target_slot].base = reinterpret_cast<void *>(0xABCD0000u);
  ctx.slots[target_slot].mapped_bytes = 8192u;
  status.unmap_base_released = false;
  status.os_resource_released = true;

  emel::io::mmap::action::effect_mark_unmap_failed_and_release_slot(runtime,
                                                                    ctx);

  CHECK(ctx.slots[target_slot].in_use);
  CHECK(ctx.slots[target_slot].base == reinterpret_cast<void *>(0xABCD0000u));
  CHECK(ctx.slots[target_slot].mapped_bytes == 8192u);
  CHECK(ctx.slots[target_slot].os_resource == -1);
  CHECK(ctx.free_count == free_count_with_slot_in_use);

  ctx.slots[target_slot].in_use = false;
  ctx.slots[target_slot].base = nullptr;
  ctx.slots[target_slot].mapped_bytes = 0u;
  ctx.slots[target_slot].os_resource = -1;
}
