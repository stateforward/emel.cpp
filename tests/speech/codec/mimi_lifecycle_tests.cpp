#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/any.hpp"

namespace {

namespace mimi = emel::speech::codec::mimi;

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  REQUIRE(size > 0);
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(stream.good());
  return bytes;
}

struct loaded_mimi_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};
};

loaded_mimi_fixture load_mimi_fixture_or_skip() {
  const auto fixture_path = repo_root() / "tests" / "models" / "mimi-tiny.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping mimi codec lifecycle test because fixture is missing: "
            << fixture_path.string());
    return {};
  }

  loaded_mimi_fixture loaded{};
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(fixture_path);

  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements = {};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>();
  const auto on_bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>();
  const auto on_bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>();
  const auto on_parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>();
  const auto on_parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>();

  const emel::gguf::loader::event::probe probe{
      std::span<const uint8_t>{loaded.file_bytes}, requirements, on_probe_done,
      on_probe_error};
  REQUIRE(loader.process_event(probe));
  loaded.kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  loaded.kv_entries.resize(requirements.kv_count);
  loaded.model->n_tensors = requirements.tensor_count;
  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{loaded.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{loaded.kv_entries},
      std::span<emel::model::data::tensor_record>{loaded.model->tensors.data(),
                                                  loaded.model->n_tensors},
      on_bind_done, on_bind_error};
  REQUIRE(loader.process_event(bind));
  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{loaded.file_bytes}, on_parse_done,
      on_parse_error};
  REQUIRE(loader.process_event(parse));

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{loaded.kv_arena},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{loaded.kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *loaded.model));

  loaded.model->name_bytes_used = 0u;
  for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
    auto &tensor = loaded.model->tensors[index];
    std::memcpy(
        loaded.model->name_storage.data() + loaded.model->name_bytes_used,
        loaded.file_bytes.data() + tensor.name_offset, tensor.name_length);
    tensor.name_offset = loaded.model->name_bytes_used;
    loaded.model->name_bytes_used += tensor.name_length;
  }
  return loaded;
}

struct codec_arenas {
  std::vector<float> prepared = {};
  std::vector<float> state = {};
  std::vector<float> workspace = {};
  std::vector<float> frame = {};
};

void size_arenas(const emel::model::data &model, codec_arenas &arenas) {
  arenas.prepared.resize(mimi::detail::required_prepared_floats(model));
  arenas.state.resize(mimi::detail::required_state_floats(model));
  arenas.workspace.resize(mimi::detail::required_workspace_floats(model));
  arenas.frame.resize(mimi::detail::required_frame_floats(model));
}

std::vector<float> deterministic_pcm(const int32_t samples) {
  std::vector<float> pcm(static_cast<size_t>(samples));
  for (int32_t index = 0; index < samples; ++index) {
    const float t = static_cast<float>(index) / 24000.0f;
    pcm[static_cast<size_t>(index)] =
        0.15f * std::sin(2.0f * 3.14159265358979323846f * 440.0f * t);
  }
  return pcm;
}

void check_session_ready(mimi::sm &machine) {
  namespace sml = stateforward::sml;
  CHECK(machine.is(sml::state<mimi::state_session_ready>));
  std::size_t visited = 0u;
  bool saw_ready = false;
  machine.visit_current_states([&](auto state) noexcept {
    ++visited;
    using state_t = typename decltype(state)::type;
    if constexpr (std::is_same_v<state_t, mimi::state_session_ready>) {
      saw_ready = true;
    }
  });
  CHECK(visited == 1u);
  CHECK(saw_ready);
}

} // namespace

TEST_CASE("mimi codec facade initializes, encodes, and decodes frames") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  codec_arenas arenas{};
  size_arenas(*loaded.model, arenas);

  mimi::sm machine{};
  namespace sml = stateforward::sml;
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));

  emel::error::type err = emel::error::cast(mimi::error::none);
  mimi::event::initialize init{*loaded.model, std::span<float>{arenas.prepared},
                               std::span<float>{arenas.state},
                               std::span<float>{arenas.workspace},
                               std::span<float>{arenas.frame}};
  init.error_out = &err;
  REQUIRE(machine.process_event(init));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(machine.frame_samples() == 1920);
  CHECK(machine.n_q() == 4);

  const auto pcm = deterministic_pcm(machine.frame_samples());
  std::vector<int32_t> codes(static_cast<size_t>(machine.n_q()), -1);
  mimi::event::encode_frame encode{std::span<const float>{pcm},
                                   std::span<int32_t>{codes}};
  encode.error_out = &err;
  REQUIRE(machine.process_event(encode));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(machine.frames_encoded() == 1u);
  for (const int32_t code : codes) {
    CHECK(code >= 0);
  }

  std::vector<float> decoded(static_cast<size_t>(machine.frame_samples()),
                             0.0f);
  mimi::event::decode_frame decode{std::span<const int32_t>{codes},
                                   std::span<float>{decoded}};
  decode.error_out = &err;
  REQUIRE(machine.process_event(decode));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(machine.frames_decoded() == 1u);

  // stream reset replays the exact same codes for the same first frame
  std::vector<int32_t> replay(codes.size(), -1);
  REQUIRE(machine.process_event(mimi::event::reset_stream{}));
  mimi::event::encode_frame encode_replay{std::span<const float>{pcm},
                                          std::span<int32_t>{replay}};
  encode_replay.error_out = &err;
  REQUIRE(machine.process_event(encode_replay));
  for (size_t index = 0; index < codes.size(); ++index) {
    CHECK(codes[index] == replay[index]);
  }
}

TEST_CASE("mimi codec facade rejects requests before initialization") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  mimi::sm machine{};
  namespace sml = stateforward::sml;

  emel::error::type err = emel::error::cast(mimi::error::none);
  std::vector<float> pcm(16, 0.0f);
  std::vector<int32_t> codes(4, -1);
  mimi::event::encode_frame encode{std::span<const float>{pcm},
                                   std::span<int32_t>{codes}};
  encode.error_out = &err;
  CHECK_FALSE(machine.process_event(encode));
  CHECK(err == emel::error::cast(mimi::error::not_initialized));
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));

  std::vector<float> decoded(16, 0.0f);
  mimi::event::decode_frame decode{std::span<const int32_t>{codes},
                                   std::span<float>{decoded}};
  decode.error_out = &err;
  CHECK_FALSE(machine.process_event(decode));
  CHECK(err == emel::error::cast(mimi::error::not_initialized));
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));
}

TEST_CASE("mimi codec facade reports bind failures explicitly") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  codec_arenas arenas{};
  size_arenas(*loaded.model, arenas);
  arenas.prepared.resize(8); // deliberately undersized

  mimi::sm machine{};
  namespace sml = stateforward::sml;
  emel::error::type err = emel::error::cast(mimi::error::none);
  mimi::event::initialize init{*loaded.model, std::span<float>{arenas.prepared},
                               std::span<float>{arenas.state},
                               std::span<float>{arenas.workspace},
                               std::span<float>{arenas.frame}};
  init.error_out = &err;
  CHECK_FALSE(machine.process_event(init));
  CHECK(err == emel::error::cast(mimi::error::bind_failed));
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));
}
