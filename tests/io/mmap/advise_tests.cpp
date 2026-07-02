#include <cstdint>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "emel/io/mmap/detail.hpp"
#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/sm.hpp"

// Coverage for the advise_mapping surface: access-pattern hints on a live
// mapping (sequential readahead, willneed prefetch, dontneed release) with the
// validation chain handle -> ownership -> range -> platform -> kind routing.

namespace {

struct map_owner_state {
  bool done = false;
  bool error = false;
  uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

struct advise_owner_state {
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

void on_advise_done(
    void *object,
    const emel::io::mmap::events::advise_mapping_done &) noexcept {
  auto *owner = static_cast<advise_owner_state *>(object);
  owner->done = true;
}

void on_advise_error(
    void *object,
    const emel::io::mmap::events::advise_mapping_error &ev) noexcept {
  auto *owner = static_cast<advise_owner_state *>(object);
  owner->error = true;
  owner->err = ev.err;
}

std::filesystem::path make_temp_file(std::string_view tag, uint64_t bytes) {
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string{"emel_io_mmap_advise_"} + std::string{tag} +
                     ".bin");
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  REQUIRE(out.good());
  const std::vector<char> payload(static_cast<size_t>(bytes), '\x5a');
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  out.close();
  return path;
}

struct mapped_fixture {
  emel::io::mmap::sm strategy{};
  map_owner_state map_owner{};
  std::filesystem::path path{};
  std::string path_str{};

  explicit mapped_fixture(std::string_view tag, uint64_t bytes = 8192u) {
    path = make_temp_file(tag, bytes);
    path_str = path.string();
    const emel::io::mmap::event::map_tensor_request request{
        .tensor_id = 71,
        .file_index = 0u,
        .file_offset = 0u,
        .byte_size = bytes,
        .file_path = path_str,
    };
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_done = {&map_owner, on_map_done};
    REQUIRE(strategy.process_event(map_request));
    REQUIRE(map_owner.done);
  }

  ~mapped_fixture() {
    emel::io::mmap::event::release_mapping release_request{71, map_owner.handle};
    (void)strategy.process_event(release_request);
    std::filesystem::remove(path);
  }
};

bool dispatch_advise(mapped_fixture &fixture, advise_owner_state &owner,
                     uint32_t handle, uint64_t offset, uint64_t length,
                     emel::io::mmap::event::advice kind,
                     int32_t tensor_id = 71) {
  emel::io::mmap::event::advise_mapping request{tensor_id, handle, offset,
                                                length, kind};
  request.on_done = {&owner, on_advise_done};
  request.on_error = {&owner, on_advise_error};
  return fixture.strategy.process_event(request);
}

} // namespace

TEST_CASE("io mmap advise applies each hint kind to a live mapping") {
  mapped_fixture fixture{"kinds"};

  advise_owner_state sequential_owner{};
  CHECK(dispatch_advise(fixture, sequential_owner, fixture.map_owner.handle, 0u,
                        8192u, emel::io::mmap::event::advice::k_sequential));
  CHECK(sequential_owner.done);
  CHECK_FALSE(sequential_owner.error);

  advise_owner_state willneed_owner{};
  CHECK(dispatch_advise(fixture, willneed_owner, fixture.map_owner.handle, 0u,
                        4096u, emel::io::mmap::event::advice::k_willneed));
  CHECK(willneed_owner.done);

  advise_owner_state dontneed_owner{};
  CHECK(dispatch_advise(fixture, dontneed_owner, fixture.map_owner.handle,
                        4096u, 4096u,
                        emel::io::mmap::event::advice::k_dontneed));
  CHECK(dontneed_owner.done);

  CHECK(fixture.strategy.is(
      stateforward::sml::state<emel::io::mmap::state_ready>));
}

TEST_CASE("io mmap advise rejects out-of-range handle") {
  mapped_fixture fixture{"bad_handle"};
  advise_owner_state owner{};

  CHECK_FALSE(dispatch_advise(fixture, owner, emel::io::mmap::k_max_mappings,
                              0u, 4096u,
                              emel::io::mmap::event::advice::k_willneed));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_request));
}

TEST_CASE("io mmap advise rejects slot not owned by tensor") {
  mapped_fixture fixture{"wrong_owner"};
  advise_owner_state owner{};

  CHECK_FALSE(dispatch_advise(fixture, owner, fixture.map_owner.handle, 0u,
                              4096u,
                              emel::io::mmap::event::advice::k_willneed,
                              /*tensor_id=*/999));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_request));
}

TEST_CASE("io mmap advise rejects unused slot handle") {
  mapped_fixture fixture{"unused_slot"};
  advise_owner_state owner{};

  // A valid-range handle whose slot was never reserved.
  const uint32_t unused_handle = fixture.map_owner.handle + 1u;
  CHECK_FALSE(dispatch_advise(fixture, owner, unused_handle, 0u, 4096u,
                              emel::io::mmap::event::advice::k_sequential));
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_request));
}

TEST_CASE("io mmap advise rejects window outside the mapped span") {
  mapped_fixture fixture{"bad_range"};

  advise_owner_state overflow_owner{};
  CHECK_FALSE(dispatch_advise(fixture, overflow_owner,
                              fixture.map_owner.handle, 4096u, 8192u,
                              emel::io::mmap::event::advice::k_willneed));
  CHECK(overflow_owner.error);
  CHECK(overflow_owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_advise_range));

  advise_owner_state empty_owner{};
  CHECK_FALSE(dispatch_advise(fixture, empty_owner, fixture.map_owner.handle,
                              0u, 0u,
                              emel::io::mmap::event::advice::k_willneed));
  CHECK(empty_owner.error);
  CHECK(empty_owner.err ==
        emel::error::cast(emel::io::mmap::error::invalid_advise_range));
}

TEST_CASE("io mmap advise reports status without callbacks") {
  mapped_fixture fixture{"no_callbacks"};

  emel::io::mmap::event::advise_mapping request{
      71, fixture.map_owner.handle, 0u, 4096u,
      emel::io::mmap::event::advice::k_sequential};
  CHECK(fixture.strategy.process_event(request));

  emel::io::mmap::event::advise_mapping bad_request{
      71, emel::io::mmap::k_max_mappings, 0u, 4096u,
      emel::io::mmap::event::advice::k_sequential};
  CHECK_FALSE(fixture.strategy.process_event(bad_request));

  CHECK(fixture.strategy.is(
      stateforward::sml::state<emel::io::mmap::state_ready>));
}
