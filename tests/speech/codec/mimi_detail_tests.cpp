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
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/detail.hpp"

namespace {

namespace codec = emel::speech::codec::mimi::detail;

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
    MESSAGE("skipping mimi codec test because fixture is missing: "
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

struct bound_codec {
  std::vector<float> prepared = {};
  std::vector<float> state = {};
  std::vector<float> workspace = {};
  std::vector<float> frame = {};
  codec::codec_runtime runtime = {};
  codec::codec_streaming_state streaming = {};
};

bool bind_or_fail(const emel::model::data &model, bound_codec &out) {
  const uint64_t prepared_floats = codec::required_prepared_floats(model);
  const uint64_t state_floats = codec::required_state_floats(model);
  const uint64_t workspace_floats = codec::required_workspace_floats(model);
  const uint64_t frame_floats = codec::required_frame_floats(model);
  REQUIRE(prepared_floats > 0u);
  REQUIRE(state_floats > 0u);
  REQUIRE(workspace_floats > 0u);
  REQUIRE(frame_floats > 0u);
  out.prepared.resize(prepared_floats);
  out.state.resize(state_floats);
  out.workspace.resize(workspace_floats);
  out.frame.resize(frame_floats);
  return codec::bind_codec_runtime(model, std::span<float>{out.prepared},
                                   std::span<float>{out.state}, out.runtime,
                                   out.streaming);
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

bool all_finite(const float *data, const size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (!std::isfinite(data[index])) {
      return false;
    }
  }
  return true;
}

// Runs the full encode chain for one frame: seanet -> transformer ->
// downsample -> rvq encode.
bool encode_frame(bound_codec &codec_state, std::span<const float> pcm,
                  std::span<int32_t> codes_out) {
  auto &runtime = codec_state.runtime;
  codec::frame_buffer io{codec_state.frame.data(), 1, runtime.frame_samples};
  std::copy(pcm.begin(), pcm.end(), codec_state.frame.begin());
  const std::span<float> workspace{codec_state.workspace};
  if (!codec::compute_seanet_stack<false>(
          runtime,
          std::span<const codec::seanet_layer_weights>{runtime.encoder_layers},
          codec_state.streaming, io, workspace)) {
    return false;
  }
  if (io.channels != runtime.dim ||
      io.length != runtime.encoder_transformer.frame_tokens) {
    return false;
  }
  if (!codec::compute_transformer<false>(
          runtime, runtime.encoder_transformer, codec_state.streaming,
          codec_state.streaming.encoder_positions, io, workspace)) {
    return false;
  }
  if (!codec::compute_streaming_conv<false>(
          runtime, runtime.downsample, codec_state.streaming, io, workspace)) {
    return false;
  }
  if (io.length != 1 || io.channels != runtime.dim) {
    return false;
  }
  return codec::compute_rvq_encode<false, false>(runtime, io, codes_out,
                                                 workspace);
}

// Runs the full decode chain for one frame of codes.
bool decode_frame(bound_codec &codec_state, std::span<const int32_t> codes,
                  std::span<float> pcm_out) {
  auto &runtime = codec_state.runtime;
  codec::frame_buffer io{codec_state.frame.data(), runtime.dim, 1};
  const std::span<float> workspace{codec_state.workspace};
  if (!codec::compute_rvq_decode<false, false>(runtime, codes, 1, io,
                                               workspace)) {
    return false;
  }
  if (!codec::compute_streaming_conv_transpose_depthwise(
          runtime, runtime.upsample, codec_state.streaming, io, workspace)) {
    return false;
  }
  if (!codec::compute_transformer<false>(
          runtime, runtime.decoder_transformer, codec_state.streaming,
          codec_state.streaming.decoder_positions, io, workspace)) {
    return false;
  }
  if (!codec::compute_seanet_stack<false>(
          runtime,
          std::span<const codec::seanet_layer_weights>{runtime.decoder_layers},
          codec_state.streaming, io, workspace)) {
    return false;
  }
  if (io.channels != 1 || io.length != runtime.frame_samples) {
    return false;
  }
  std::copy_n(io.data, static_cast<size_t>(io.length), pcm_out.begin());
  return true;
}

} // namespace

TEST_CASE("mimi codec binds and encodes deterministic frames") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  bound_codec codec_state{};
  REQUIRE(bind_or_fail(*loaded.model, codec_state));
  CHECK(codec_state.runtime.frame_samples == 1920);
  CHECK(codec_state.runtime.n_q == 4);
  CHECK(codec_state.runtime.dim == 16);

  const auto pcm = deterministic_pcm(codec_state.runtime.frame_samples);
  std::vector<int32_t> codes(static_cast<size_t>(codec_state.runtime.n_q), -1);
  REQUIRE(encode_frame(codec_state, pcm, std::span<int32_t>{codes}));
  for (const int32_t code : codes) {
    CHECK(code >= 0);
    CHECK(code < codec_state.runtime.quantizer.codebook_entries);
  }

  // second frame advances streaming state and stays finite
  std::vector<int32_t> second_codes(codes.size(), -1);
  REQUIRE(encode_frame(codec_state, pcm, std::span<int32_t>{second_codes}));

  // resetting the stream reproduces the first frame exactly
  codec::reset_streaming_state(codec_state.runtime, codec_state.streaming);
  std::vector<int32_t> replay_codes(codes.size(), -1);
  REQUIRE(encode_frame(codec_state, pcm, std::span<int32_t>{replay_codes}));
  for (size_t index = 0; index < codes.size(); ++index) {
    CHECK(codes[index] == replay_codes[index]);
  }
}

TEST_CASE("mimi codec decodes codes back to a full frame of audio") {
  auto loaded = load_mimi_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  bound_codec codec_state{};
  REQUIRE(bind_or_fail(*loaded.model, codec_state));

  std::vector<int32_t> codes(static_cast<size_t>(codec_state.runtime.n_q));
  for (size_t index = 0; index < codes.size(); ++index) {
    codes[index] = static_cast<int32_t>(
        (index * 7 + 3) %
        static_cast<size_t>(codec_state.runtime.quantizer.codebook_entries));
  }
  std::vector<float> pcm(static_cast<size_t>(codec_state.runtime.frame_samples),
                         0.0f);
  REQUIRE(decode_frame(codec_state, std::span<const int32_t>{codes},
                       std::span<float>{pcm}));
  CHECK(all_finite(pcm.data(), pcm.size()));

  // out-of-range codes are rejected explicitly
  std::vector<int32_t> bad_codes = codes;
  bad_codes[0] = codec_state.runtime.quantizer.codebook_entries;
  codec::frame_buffer io{codec_state.frame.data(), codec_state.runtime.dim, 1};
  CHECK_FALSE(codec::compute_rvq_decode<false, false>(
      codec_state.runtime, std::span<const int32_t>{bad_codes}, 1, io,
      std::span<float>{codec_state.workspace}));
}
