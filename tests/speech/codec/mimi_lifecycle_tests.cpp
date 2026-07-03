#include "doctest/doctest.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST_CASE("mimi codec initialize rejects out-of-contract model metadata") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  codec_arenas arenas{};
  size_arenas(*loaded.model, arenas);

  namespace sml = stateforward::sml;
  const auto expect_bind_failed = [&](emel::model::data &model) {
    mimi::sm machine{};
    emel::error::type err = emel::error::cast(mimi::error::none);
    g_init_error = emel::error::cast(mimi::error::none);
    mimi::event::initialize init{model, std::span<float>{arenas.prepared},
                                 std::span<float>{arenas.state},
                                 std::span<float>{arenas.workspace},
                                 std::span<float>{arenas.frame}};
    init.error_out = &err;
    init.on_error = emel::callback<void(
        const mimi::events::initialize_error &)>::from<&on_initialize_error>();
    CHECK_FALSE(machine.process_event(init));
    CHECK(err == emel::error::cast(mimi::error::bind_failed));
    CHECK(g_init_error == emel::error::cast(mimi::error::bind_failed));
    CHECK(machine.is(sml::state<mimi::state_uninitialized>));
  };

  SUBCASE("transformer layer count beyond the fixed runtime array") {
    loaded.model->mimi.transformer_num_layers = 17;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("rvq split beyond the fixed level arrays") {
    loaded.model->mimi.n_q = 40;
    loaded.model->mimi.semantic_n_q = 33;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("odd attention head size") {
    // heads == dim makes head_dim 1, which the fp16 rotary halving cannot
    // serve; validation must reject it instead of silently skipping the
    // transformer at compute time.
    loaded.model->mimi.transformer_num_heads =
        static_cast<int32_t>(loaded.model->mimi.dim);
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("tensor metadata without bound weight storage") {
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      loaded.model->tensors[index].data = nullptr;
      loaded.model->tensors[index].data_size = 0u;
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("transposed conv kernel shorter than its stride") {
    // The decode overlap-add carries taps - stride samples of state; a
    // one-tap upsample kernel against stride 2 would index outside it.
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      auto &tensor = loaded.model->tensors[index];
      const std::string_view name{
          loaded.model->name_storage.data() + tensor.name_offset,
          tensor.name_length};
      if (name == "mimi.upsample.convtr.convtr.convtr.weight") {
        tensor.dims[0] = 1;
      }
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("seanet transposed layer kernel shorter than its stride") {
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      auto &tensor = loaded.model->tensors[index];
      const std::string_view name{
          loaded.model->name_storage.data() + tensor.name_offset,
          tensor.name_length};
      if (name.find("decoder.model.") != std::string_view::npos &&
          name.find(".convtr.convtr.weight") != std::string_view::npos) {
        tensor.dims[0] = 1;
      }
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("zero transformer context") {
    // compute_transformer takes position % context; a zero context would
    // divide by zero on the first frame instead of failing at bind.
    loaded.model->mimi.transformer_context = 0;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("empty rvq codebook dimensions") {
    loaded.model->mimi.codebook_dim = 0;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("empty rvq codebook entries") {
    loaded.model->mimi.card = 0;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("seanet streaming conv kernel shorter than its stride") {
    // Normal (non-transposed) streaming convs carry taps - stride samples of
    // state; a strided encoder conv with a one-tap kernel would wrap that
    // span negative in the sizing accounting.
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      auto &tensor = loaded.model->tensors[index];
      const std::string_view name{
          loaded.model->name_storage.data() + tensor.name_offset,
          tensor.name_length};
      if (name.find("encoder.model.") != std::string_view::npos &&
          name.find(".conv.conv.weight") != std::string_view::npos &&
          name.find(".block.") == std::string_view::npos) {
        tensor.dims[0] = 1;
      }
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("downsample conv kernel shorter than its stride") {
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      auto &tensor = loaded.model->tensors[index];
      const std::string_view name{
          loaded.model->name_storage.data() + tensor.name_offset,
          tensor.name_length};
      if (name == "mimi.downsample.conv.conv.conv.weight") {
        tensor.dims[0] = 1;
      }
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("non-finite frame rate") {
    // frame_samples is computed as sample_rate / frame_rate and cast to
    // int32_t; the cast is undefined for NaN.
    loaded.model->mimi.frame_rate = std::numeric_limits<float>::quiet_NaN();
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("tiny frame rate overflowing the frame sample cast") {
    loaded.model->mimi.frame_rate = 1.0e-30f;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("non-positive rotary max period") {
    // compute_transformer evaluates log(max_period).
    loaded.model->mimi.transformer_max_period = 0;
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("tensor rank beyond the stored dimensions") {
    // tensor_record carries four extents; a larger declared rank must reject
    // instead of scanning past the dims array in the element count.
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      loaded.model->tensors[index].n_dims = 8;
    }
    expect_bind_failed(*loaded.model);
  }

  SUBCASE("tensor storage smaller than its metadata") {
    // The prepare copies read the full dtype payload; a one-byte buffer with
    // matching element metadata must fail validation instead of reading past
    // its storage.
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      loaded.model->tensors[index].data_size = 1u;
    }
    expect_bind_failed(*loaded.model);
  }
}

TEST_CASE("mimi codec facade reports unexpected event ordering as errors") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  codec_arenas arenas{};
  size_arenas(*loaded.model, arenas);

  namespace sml = stateforward::sml;
  mimi::sm machine{};

  // reset_stream before initialization is a caller ordering error, not a
  // silent success.
  CHECK_FALSE(machine.process_event(mimi::event::reset_stream{}));
  CHECK(machine.is(sml::state<mimi::state_uninitialized>));

  emel::error::type err = emel::error::cast(mimi::error::none);
  mimi::event::initialize init{*loaded.model, std::span<float>{arenas.prepared},
                               std::span<float>{arenas.state},
                               std::span<float>{arenas.workspace},
                               std::span<float>{arenas.frame}};
  init.error_out = &err;
  REQUIRE(machine.process_event(init));
  check_session_ready(machine);

  // initialize while a session is ready is equally out of order; the
  // dispatch must fail and surface the explicit unexpected_event error.
  emel::error::type second_err = emel::error::cast(mimi::error::none);
  mimi::event::initialize second{
      *loaded.model, std::span<float>{arenas.prepared},
      std::span<float>{arenas.state}, std::span<float>{arenas.workspace},
      std::span<float>{arenas.frame}};
  second.error_out = &second_err;
  CHECK_FALSE(machine.process_event(second));
  CHECK(second_err == emel::error::cast(mimi::error::unexpected_event));
  check_session_ready(machine);

  // A well-formed reset after initialization still succeeds.
  CHECK(machine.process_event(mimi::event::reset_stream{}));
  check_session_ready(machine);
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

TEST_CASE("mimi child actors report validation errors when driven directly") {
  // The facade pre-validates every child request, so these explicit error
  // routes only execute when the child machines are driven directly (they
  // are independently usable actors): unbound runtime, request-shape, and
  // buffer-capacity guards each select their own marked error transition.
  namespace encoder = mimi::encoder;
  namespace decoder = mimi::decoder;
  namespace quantizer = mimi::quantizer;

  mimi::detail::codec_runtime unbound_runtime{};
  mimi::detail::codec_streaming_state unbound_streaming{};
  std::array<float, 8> tiny_buf{};
  std::array<int32_t, 8> tiny_codes{};

  // unbound runtime -> runtime_unbound on all three actors
  {
    encoder::sm machine{};
    emel::error::type err = emel::error::cast(encoder::error::none);
    encoder::event::encode ev{unbound_runtime,
                              unbound_streaming,
                              std::span<const float>{tiny_buf},
                              std::span<float>{tiny_buf},
                              std::span<float>{tiny_buf},
                              std::span<float>{tiny_buf}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(encoder::error::runtime_unbound));
  }
  {
    decoder::sm machine{};
    emel::error::type err = emel::error::cast(decoder::error::none);
    decoder::event::decode ev{unbound_runtime,
                              unbound_streaming,
                              std::span<const float>{tiny_buf},
                              std::span<float>{tiny_buf},
                              std::span<float>{tiny_buf},
                              std::span<float>{tiny_buf}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(decoder::error::runtime_unbound));
  }
  {
    quantizer::sm machine{};
    emel::error::type err = emel::error::cast(quantizer::error::none);
    quantizer::event::encode ev{unbound_runtime,
                                std::span<float>{tiny_buf},
                                std::span<int32_t>{tiny_codes},
                                std::span<float>{tiny_buf}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(quantizer::error::runtime_unbound));

    err = emel::error::cast(quantizer::error::none);
    quantizer::event::decode dev{unbound_runtime,
                                 std::span<const int32_t>{tiny_codes},
                                 std::span<float>{tiny_buf},
                                 std::span<float>{tiny_buf}};
    dev.error_out = &err;
    CHECK_FALSE(machine.process_event(dev));
    CHECK(err == emel::error::cast(quantizer::error::runtime_unbound));
  }

  // bound runtime: shape and capacity violations route their own errors
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }
  std::vector<float> prepared(mimi::prepared_arena_floats(*loaded.model));
  std::vector<float> state(mimi::state_arena_floats(*loaded.model));
  std::vector<float> workspace(mimi::workspace_arena_floats(*loaded.model));
  std::vector<float> frame(mimi::frame_arena_floats(*loaded.model));
  mimi::detail::codec_runtime runtime{};
  mimi::detail::codec_streaming_state streaming{};
  REQUIRE(mimi::detail::validate_codec_contract(*loaded.model));
  mimi::detail::bind_codec_runtime(*loaded.model, std::span<float>{prepared},
                                   std::span<float>{state}, runtime, streaming);
  REQUIRE(runtime.model != nullptr);

  std::vector<float> pcm(static_cast<size_t>(runtime.frame_samples), 0.0f);
  std::vector<float> latent(static_cast<size_t>(runtime.dim), 0.0f);
  std::vector<int32_t> codes(static_cast<size_t>(runtime.n_q), 0);

  {
    // pcm shorter than one frame -> request_shape
    encoder::sm machine{};
    emel::error::type err = emel::error::cast(encoder::error::none);
    encoder::event::encode ev{runtime,
                              streaming,
                              std::span<const float>{pcm.data(), pcm.size() - 1u},
                              std::span<float>{frame},
                              std::span<float>{workspace},
                              std::span<float>{latent}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(encoder::error::request_shape));

    // undersized workspace -> buffer_capacity
    err = emel::error::cast(encoder::error::none);
    encoder::event::encode cap_ev{runtime,
                                  streaming,
                                  std::span<const float>{pcm},
                                  std::span<float>{frame},
                                  std::span<float>{tiny_buf},
                                  std::span<float>{latent}};
    cap_ev.error_out = &err;
    CHECK_FALSE(machine.process_event(cap_ev));
    CHECK(err == emel::error::cast(encoder::error::buffer_capacity));
  }
  {
    // latent narrower than dim -> request_shape
    decoder::sm machine{};
    emel::error::type err = emel::error::cast(decoder::error::none);
    decoder::event::decode ev{runtime,
                              streaming,
                              std::span<const float>{latent.data(),
                                                     latent.size() - 1u},
                              std::span<float>{frame},
                              std::span<float>{workspace},
                              std::span<float>{pcm}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(decoder::error::request_shape));

    // undersized frame staging -> buffer_capacity
    err = emel::error::cast(decoder::error::none);
    decoder::event::decode cap_ev{runtime,
                                  streaming,
                                  std::span<const float>{latent},
                                  std::span<float>{tiny_buf},
                                  std::span<float>{workspace},
                                  std::span<float>{pcm}};
    cap_ev.error_out = &err;
    CHECK_FALSE(machine.process_event(cap_ev));
    CHECK(err == emel::error::cast(decoder::error::buffer_capacity));
  }
  {
    // latent narrower than dim -> request_shape; out-of-range decode code ->
    // code_range
    quantizer::sm machine{};
    emel::error::type err = emel::error::cast(quantizer::error::none);
    quantizer::event::encode ev{runtime,
                                std::span<float>{latent.data(),
                                                 latent.size() - 1u},
                                std::span<int32_t>{codes},
                                std::span<float>{workspace}};
    ev.error_out = &err;
    CHECK_FALSE(machine.process_event(ev));
    CHECK(err == emel::error::cast(quantizer::error::request_shape));

    err = emel::error::cast(quantizer::error::none);
    std::vector<int32_t> bad_codes(codes.size(), 0);
    bad_codes[0] = runtime.quantizer.codebook_entries;
    quantizer::event::decode dev{runtime,
                                 std::span<const int32_t>{bad_codes},
                                 std::span<float>{latent},
                                 std::span<float>{workspace}};
    dev.error_out = &err;
    CHECK_FALSE(machine.process_event(dev));
    CHECK(err == emel::error::cast(quantizer::error::code_range));
  }
}
