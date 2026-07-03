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

// modeled-event observation: initialize_done carries the codec dims, and
// the done callbacks count completed frames (no context peeking)
int32_t g_frame_samples = 0;
int32_t g_n_q = 0;
uint32_t g_encode_done_count = 0;
uint32_t g_decode_done_count = 0;
void on_initialize_done(const mimi::events::initialize_done &done) {
  g_frame_samples = done.frame_samples;
  g_n_q = done.n_q;
}
void on_encode_frame_done(const mimi::events::encode_frame_done &) {
  ++g_encode_done_count;
}
void on_decode_frame_done(const mimi::events::decode_frame_done &) {
  ++g_decode_done_count;
}
emel::error::type g_init_error = 0;
uint32_t g_encode_error_count = 0;
uint32_t g_decode_error_count = 0;
void on_initialize_error(const mimi::events::initialize_error &error_ev) {
  g_init_error = error_ev.err;
}
void on_encode_frame_error(const mimi::events::encode_frame_error &) {
  ++g_encode_error_count;
}
void on_decode_frame_error(const mimi::events::decode_frame_error &) {
  ++g_decode_error_count;
}

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

loaded_mimi_fixture
load_mimi_fixture_or_skip(const char *fixture_name = "mimi-tiny.gguf") {
  const auto fixture_path = repo_root() / "tests" / "models" / fixture_name;
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
  arenas.prepared.resize(mimi::prepared_arena_floats(model));
  arenas.state.resize(mimi::state_arena_floats(model));
  arenas.workspace.resize(mimi::workspace_arena_floats(model));
  arenas.frame.resize(mimi::frame_arena_floats(model));
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
  g_frame_samples = 0;
  g_n_q = 0;
  g_encode_done_count = 0;
  g_decode_done_count = 0;
  mimi::event::initialize init{*loaded.model, std::span<float>{arenas.prepared},
                               std::span<float>{arenas.state},
                               std::span<float>{arenas.workspace},
                               std::span<float>{arenas.frame}};
  init.error_out = &err;
  init.on_done = emel::callback<void(
      const mimi::events::initialize_done &)>::from<&on_initialize_done>();
  REQUIRE(machine.process_event(init));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(g_frame_samples == 1920);
  CHECK(g_n_q == 4);

  const auto pcm = deterministic_pcm(g_frame_samples);
  std::vector<int32_t> codes(static_cast<size_t>(g_n_q), -1);
  mimi::event::encode_frame encode{std::span<const float>{pcm},
                                   std::span<int32_t>{codes}};
  encode.error_out = &err;
  encode.on_done = emel::callback<void(
      const mimi::events::encode_frame_done &)>::from<&on_encode_frame_done>();
  REQUIRE(machine.process_event(encode));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(g_encode_done_count == 1u);
  for (const int32_t code : codes) {
    CHECK(code >= 0);
  }

  std::vector<float> decoded(static_cast<size_t>(g_frame_samples), 0.0f);
  mimi::event::decode_frame decode{std::span<const int32_t>{codes},
                                   std::span<float>{decoded}};
  decode.error_out = &err;
  decode.on_done = emel::callback<void(
      const mimi::events::decode_frame_done &)>::from<&on_decode_frame_done>();
  REQUIRE(machine.process_event(decode));
  CHECK(err == emel::error::cast(mimi::error::none));
  check_session_ready(machine);
  CHECK(g_decode_done_count == 1u);

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
  g_init_error = emel::error::cast(mimi::error::none);
  init.on_error = emel::callback<void(
      const mimi::events::initialize_error &)>::from<&on_initialize_error>();
  CHECK_FALSE(machine.process_event(init));
  CHECK(err == emel::error::cast(mimi::error::arena_capacity));
  CHECK(g_init_error == emel::error::cast(mimi::error::arena_capacity));
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));

  // A model that does not satisfy the mimi contract (wrong component) routes
  // the distinct bind_failed error through the contract guard.
  auto lm_loaded = load_mimi_fixture_or_skip("moshi-tiny-lm.gguf");
  if (lm_loaded.model != nullptr) {
    codec_arenas lm_arenas{};
    lm_arenas.prepared.resize(64);
    lm_arenas.state.resize(64);
    lm_arenas.workspace.resize(64);
    lm_arenas.frame.resize(64);
    mimi::sm lm_machine{};
    err = emel::error::cast(mimi::error::none);
    g_init_error = emel::error::cast(mimi::error::none);
    mimi::event::initialize lm_init{*lm_loaded.model,
                                    std::span<float>{lm_arenas.prepared},
                                    std::span<float>{lm_arenas.state},
                                    std::span<float>{lm_arenas.workspace},
                                    std::span<float>{lm_arenas.frame}};
    lm_init.error_out = &err;
    lm_init.on_error = emel::callback<void(
        const mimi::events::initialize_error &)>::from<&on_initialize_error>();
    CHECK_FALSE(lm_machine.process_event(lm_init));
    CHECK(err == emel::error::cast(mimi::error::bind_failed));
    CHECK(g_init_error == emel::error::cast(mimi::error::bind_failed));
    CHECK(lm_machine.is(sml::state<mimi::state_uninitialized>));
  }
}

TEST_CASE("mimi codec facade rejects malformed frame requests explicitly") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  codec_arenas arenas{};
  size_arenas(*loaded.model, arenas);

  mimi::sm machine{};
  namespace sml = stateforward::sml;
  emel::error::type err = emel::error::cast(mimi::error::none);
  g_frame_samples = 0;
  g_n_q = 0;
  mimi::event::initialize init{*loaded.model, std::span<float>{arenas.prepared},
                               std::span<float>{arenas.state},
                               std::span<float>{arenas.workspace},
                               std::span<float>{arenas.frame}};
  init.error_out = &err;
  init.on_done = emel::callback<void(
      const mimi::events::initialize_done &)>::from<&on_initialize_done>();
  REQUIRE(machine.process_event(init));
  REQUIRE(g_frame_samples > 0);
  REQUIRE(g_n_q > 0);

  // encode: wrong PCM length routes the explicit request_shape error
  std::vector<float> short_pcm(static_cast<size_t>(g_frame_samples) - 1u, 0.0f);
  std::vector<int32_t> codes(static_cast<size_t>(g_n_q), -1);
  mimi::event::encode_frame bad_encode{std::span<const float>{short_pcm},
                                       std::span<int32_t>{codes}};
  bad_encode.error_out = &err;
  g_encode_error_count = 0;
  bad_encode.on_error = emel::callback<void(
      const mimi::events::encode_frame_error &)>::from<&on_encode_frame_error>();
  CHECK_FALSE(machine.process_event(bad_encode));
  CHECK(g_encode_error_count == 1u);
  CHECK(err == emel::error::cast(mimi::error::request_shape));
  CHECK(machine.is(sml::state<mimi::state_session_ready>));

  // decode: undersized output routes request_shape
  std::vector<int32_t> valid_codes(static_cast<size_t>(g_n_q), 0);
  std::vector<float> short_pcm_out(static_cast<size_t>(g_frame_samples) - 1u);
  mimi::event::decode_frame bad_decode{std::span<const int32_t>{valid_codes},
                                       std::span<float>{short_pcm_out}};
  bad_decode.error_out = &err;
  g_decode_error_count = 0;
  bad_decode.on_error = emel::callback<void(
      const mimi::events::decode_frame_error &)>::from<&on_decode_frame_error>();
  CHECK_FALSE(machine.process_event(bad_decode));
  CHECK(g_decode_error_count == 1u);
  CHECK(err == emel::error::cast(mimi::error::request_shape));
  CHECK(machine.is(sml::state<mimi::state_session_ready>));

  // decode: an out-of-range code routes the explicit code_range error
  std::vector<int32_t> bad_codes(static_cast<size_t>(g_n_q), 0);
  bad_codes[0] = 1 << 20;
  std::vector<float> pcm_out(static_cast<size_t>(g_frame_samples), 0.0f);
  mimi::event::decode_frame range_decode{std::span<const int32_t>{bad_codes},
                                         std::span<float>{pcm_out}};
  range_decode.error_out = &err;
  CHECK_FALSE(machine.process_event(range_decode));
  CHECK(err == emel::error::cast(mimi::error::code_range));
  CHECK(machine.is(sml::state<mimi::state_session_ready>));

  // the session stays usable after rejected requests
  const auto pcm = deterministic_pcm(g_frame_samples);
  mimi::event::encode_frame good_encode{std::span<const float>{pcm},
                                        std::span<int32_t>{codes}};
  good_encode.error_out = &err;
  CHECK(machine.process_event(good_encode));
  CHECK(err == emel::error::cast(mimi::error::none));
}

TEST_CASE("mimi codec facade streams the f16 and q8 operand-class fixtures") {
  // One committed fixture per bound operand class: mimi-tiny-f16.gguf stores
  // conv/projection weights f16 (guard_conv_f16 rows, f16 im2col + mul_mat
  // paths, raw-f16 RVQ projections); mimi-tiny-q8.gguf carries q8_0
  // transformer/RVQ projections (guard_proj_q8 / guard_class_q8 rows, q8
  // matvec paths). The streaming contract is identical across classes:
  // deterministic codes, exact replay after reset_stream.
  for (const char *fixture_name : {"mimi-tiny-f16.gguf", "mimi-tiny-q8.gguf"}) {
    CAPTURE(fixture_name);
    auto loaded = load_mimi_fixture_or_skip(fixture_name);
    if (loaded.model == nullptr) {
      continue;
    }

    codec_arenas arenas{};
    size_arenas(*loaded.model, arenas);

    mimi::sm machine{};
    namespace sml = stateforward::sml;
    emel::error::type err = emel::error::cast(mimi::error::none);
    g_frame_samples = 0;
    g_n_q = 0;
    mimi::event::initialize init{*loaded.model,
                                 std::span<float>{arenas.prepared},
                                 std::span<float>{arenas.state},
                                 std::span<float>{arenas.workspace},
                                 std::span<float>{arenas.frame}};
    init.error_out = &err;
    init.on_done = emel::callback<void(
        const mimi::events::initialize_done &)>::from<&on_initialize_done>();
    REQUIRE(machine.process_event(init));
    REQUIRE(err == emel::error::cast(mimi::error::none));
    REQUIRE(g_frame_samples > 0);
    REQUIRE(g_n_q > 0);

    const auto pcm = deterministic_pcm(g_frame_samples);
    std::vector<int32_t> codes(static_cast<size_t>(g_n_q), -1);
    mimi::event::encode_frame encode{std::span<const float>{pcm},
                                     std::span<int32_t>{codes}};
    encode.error_out = &err;
    REQUIRE(machine.process_event(encode));
    CHECK(err == emel::error::cast(mimi::error::none));
    for (const int32_t code : codes) {
      CHECK(code >= 0);
    }

    // second frame advances streaming state
    std::vector<int32_t> second(codes.size(), -1);
    mimi::event::encode_frame encode_second{std::span<const float>{pcm},
                                            std::span<int32_t>{second}};
    encode_second.error_out = &err;
    REQUIRE(machine.process_event(encode_second));

    std::vector<float> decoded(static_cast<size_t>(g_frame_samples), 0.0f);
    mimi::event::decode_frame decode{std::span<const int32_t>{codes},
                                     std::span<float>{decoded}};
    decode.error_out = &err;
    REQUIRE(machine.process_event(decode));
    CHECK(err == emel::error::cast(mimi::error::none));
    bool all_finite = true;
    for (const float sample : decoded) {
      all_finite = all_finite && std::isfinite(sample);
    }
    CHECK(all_finite);

    // reset replays the first frame exactly
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
}
