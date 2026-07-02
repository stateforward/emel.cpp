#include "bench_cases.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/any.hpp"

// speech_codec_mimi suite: EMEL-lane frame latency for the Mimi codec facade
// against the committed tiny fixture (tests/models/mimi-tiny.gguf). The
// budget of interest is the 80 ms real-time frame period. The reference lane
// is subprocess-driven via reference_backends/moshi_cpp_mimi.json (two-lane
// isolation), so no in-process reference cases are appended here.
namespace {

namespace mimi = emel::speech::codec::mimi;

int32_t g_frame_samples = 0;
int32_t g_n_q = 0;
void on_codec_initialized(const mimi::events::initialize_done &done) {
  g_frame_samples = done.frame_samples;
  g_n_q = done.n_q;
}

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

struct codec_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};
  std::vector<float> prepared = {};
  std::vector<float> state = {};
  std::vector<float> workspace = {};
  std::vector<float> frame = {};
  std::unique_ptr<mimi::sm> codec = {};
  std::vector<float> pcm = {};
  std::vector<int32_t> codes = {};
  std::vector<float> decoded = {};
};

bool load_fixture(codec_fixture &fixture) {
  const auto path = std::filesystem::path{EMEL_BENCH_REPO_ROOT} / "tests" /
                    "models" / "mimi-tiny.gguf";
  std::ifstream stream(path, std::ios::binary);
  if (!stream.good()) {
    return false;
  }
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  fixture.file_bytes.resize(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(fixture.file_bytes.data()), size);
  if (!stream.good()) {
    return false;
  }
  fixture.model = std::make_unique<emel::model::data>();

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
      std::span<const uint8_t>{fixture.file_bytes}, requirements, on_probe_done,
      on_probe_error};
  if (!loader.process_event(probe)) {
    return false;
  }
  fixture.kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.model->n_tensors = requirements.tensor_count;
  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{fixture.model->tensors.data(),
                                                  fixture.model->n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    return false;
  }
  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{fixture.file_bytes}, on_parse_done,
      on_parse_error};
  if (!loader.process_event(parse)) {
    return false;
  }

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries},
  };
  if (!emel::model::detail::load_hparams_from_gguf(binding, *fixture.model)) {
    return false;
  }
  fixture.model->name_bytes_used = 0u;
  for (uint32_t index = 0u; index < fixture.model->n_tensors; ++index) {
    auto &tensor = fixture.model->tensors[index];
    std::memcpy(
        fixture.model->name_storage.data() + fixture.model->name_bytes_used,
        fixture.file_bytes.data() + tensor.name_offset, tensor.name_length);
    tensor.name_offset = fixture.model->name_bytes_used;
    fixture.model->name_bytes_used += tensor.name_length;
  }

  fixture.prepared.resize(
      mimi::detail::required_prepared_floats(*fixture.model));
  fixture.state.resize(mimi::detail::required_state_floats(*fixture.model));
  fixture.workspace.resize(
      mimi::detail::required_workspace_floats(*fixture.model));
  fixture.frame.resize(mimi::detail::required_frame_floats(*fixture.model));
  fixture.codec = std::make_unique<mimi::sm>();
  mimi::event::initialize init{
      *fixture.model, std::span<float>{fixture.prepared},
      std::span<float>{fixture.state}, std::span<float>{fixture.workspace},
      std::span<float>{fixture.frame}};
  init.on_done = emel::callback<void(
      const mimi::events::initialize_done &)>::from<&on_codec_initialized>();
  if (!fixture.codec->process_event(init)) {
    return false;
  }
  const auto samples = static_cast<size_t>(g_frame_samples);
  fixture.pcm.resize(samples);
  for (size_t index = 0; index < samples; ++index) {
    fixture.pcm[index] =
        0.15f * static_cast<float>((index % 55u)) / 55.0f - 0.075f;
  }
  fixture.codes.assign(static_cast<size_t>(g_n_q), 0);
  fixture.decoded.assign(samples, 0.0f);
  return true;
}

} // namespace

namespace emel::bench {

void append_emel_speech_codec_mimi_cases(std::vector<result> &results,
                                         const config &cfg) {
  codec_fixture fixture{};
  if (!load_fixture(fixture)) {
    std::fprintf(stderr, "error: speech_codec_mimi bench fixture setup failed "
                         "(tests/models/mimi-tiny.gguf)\n");
    std::exit(1);
  }

  {
    auto fn = [&]() {
      mimi::event::encode_frame ev{std::span<const float>{fixture.pcm},
                                   std::span<int32_t>{fixture.codes}};
      (void)fixture.codec->process_event(ev);
    };
    results.push_back(
        measure_case("speech_codec_mimi/encode_frame_tiny", cfg, fn));
  }

  {
    auto fn = [&]() {
      mimi::event::decode_frame ev{std::span<const int32_t>{fixture.codes},
                                   std::span<float>{fixture.decoded}};
      (void)fixture.codec->process_event(ev);
    };
    results.push_back(
        measure_case("speech_codec_mimi/decode_frame_tiny", cfg, fn));
  }

  {
    auto fn = [&]() {
      (void)fixture.codec->process_event(mimi::event::reset_stream{});
      mimi::event::encode_frame encode_ev{std::span<const float>{fixture.pcm},
                                          std::span<int32_t>{fixture.codes}};
      (void)fixture.codec->process_event(encode_ev);
      mimi::event::decode_frame decode_ev{
          std::span<const int32_t>{fixture.codes},
          std::span<float>{fixture.decoded}};
      (void)fixture.codec->process_event(decode_ev);
    };
    results.push_back(
        measure_case("speech_codec_mimi/roundtrip_frame_tiny", cfg, fn));
  }
}

void append_reference_speech_codec_mimi_cases(std::vector<result> &results,
                                              const config &cfg) {
  // The moshi.cpp reference lane runs as a separate executable
  // (moshi_reference_driver) via reference_backends/moshi_cpp_mimi.json.
  (void)results;
  (void)cfg;
}

} // namespace emel::bench
