#include "doctest/doctest.h"

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
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/whisper/detail.hpp"
#include "emel/speech/encoder/whisper/detail.hpp"
#include "emel/speech/encoder/whisper/sm.hpp"
#include "emel/speech/recognizer/sm.hpp"
#include "emel/speech/recognizer_routes/whisper/any.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace {

namespace encoder = emel::speech::encoder::whisper;

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

struct encoder_done_capture {
  int32_t calls = 0;
  int32_t frame_count = 0;
  int32_t width = 0;
  uint64_t digest = 0;
};

struct encoder_error_capture {
  int32_t calls = 0;
  emel::error::type err = emel::error::cast(encoder::error::none);
};

void record_encoder_done(encoder_done_capture &capture,
                         const encoder::events::encode_done &done) noexcept {
  ++capture.calls;
  capture.frame_count = done.frame_count;
  capture.width = done.width;
  capture.digest = done.digest;
}

void record_encoder_error(encoder_error_capture &capture,
                          const encoder::events::encode_error &error) noexcept {
  ++capture.calls;
  capture.err = error.err;
}

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path whisper_fixture_path() {
  return repo_root() / "tests" / "models" / "model-tiny-q80.gguf";
}

std::filesystem::path whisper_tokenizer_path() {
  return repo_root() / "tests" / "models" / "tokenizer-tiny.json";
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

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  REQUIRE(size > 0);
  stream.seekg(0, std::ios::beg);
  std::string text(static_cast<size_t>(size), '\0');
  stream.read(text.data(), size);
  REQUIRE(stream.good());
  return text;
}

void materialize_tensor_names_from_file(
    emel::model::data &model, const std::vector<uint8_t> &file_bytes) {
  model.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= file_bytes.size());
    REQUIRE(static_cast<size_t>(model.name_bytes_used) + length <=
            model.name_storage.size());
    std::memcpy(model.name_storage.data() + model.name_bytes_used,
                file_bytes.data() + source_offset, length);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
}

bool load_whisper_fixture_binding(
    const std::vector<uint8_t> &file_bytes, std::vector<uint8_t> &kv_arena,
    std::vector<emel::gguf::loader::kv_entry> &kv_entries,
    emel::model::data &model_out) {
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
      std::span<const uint8_t>{file_bytes}, requirements, on_probe_done,
      on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > model_out.tensors.size()) {
    return false;
  }

  kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  kv_entries.resize(requirements.kv_count);
  model_out.n_tensors = requirements.tensor_count;

  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{kv_arena},
      std::span<emel::gguf::loader::kv_entry>{kv_entries},
      std::span<emel::model::data::tensor_record>{model_out.tensors.data(),
                                                  model_out.n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    return false;
  }

  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes}, on_parse_done, on_parse_error};
  return loader.process_event(parse);
}

struct loaded_whisper_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};
  encoder::detail::execution_contract contract = {};
};

void mark_encoder_aux_tensors_f32(emel::model::data &model) {
  for (uint32_t index = 0; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const auto name = emel::model::tensor_name_view(model, tensor);
    if (name.starts_with("model.encoder.") &&
        (tensor.n_dims == 1 ||
         name == "model.encoder.embed_positions.weight")) {
      tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
    }
  }
}

loaded_whisper_fixture load_fixture_or_skip() {
  const auto fixture_path = whisper_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping Whisper encoder fixture test because fixture is missing: "
            << fixture_path.string());
    return {};
  }

  loaded_whisper_fixture loaded{};
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(fixture_path);
  REQUIRE(load_whisper_fixture_binding(loaded.file_bytes, loaded.kv_arena,
                                       loaded.kv_entries, *loaded.model));
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{loaded.kv_arena},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{loaded.kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *loaded.model));
  loaded.model->weights_data = loaded.file_bytes.data();
  loaded.model->weights_size = loaded.file_bytes.size();
  loaded.model->weights_mapped = true;
  materialize_tensor_names_from_file(*loaded.model, loaded.file_bytes);
  emel::model::whisper::detail::execution_contract model_contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(
              *loaded.model, model_contract) ==
          emel::error::cast(emel::model::loader::error::none));
  loaded.contract = encoder::detail::bind_execution_contract(*loaded.model);
  return loaded;
}

std::vector<float> deterministic_pcm(const size_t sample_count) {
  std::vector<float> pcm(sample_count);
  for (size_t index = 0; index < sample_count; ++index) {
    const float t = static_cast<float>(index) / 16000.0f;
    pcm[index] = 0.15f * std::sin(2.0f * 3.14159265358979323846f * 440.0f * t);
  }
  return pcm;
}

} // namespace

TEST_CASE("whisper_encoder_rejects_invalid_audio_contracts") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  std::vector<float> pcm = deterministic_pcm(320);
  std::vector<float> workspace(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_workspace_floats(
          pcm.size())));
  std::vector<float> output(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_encoder_output_floats(
          pcm.size())));
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t digest = 0;
  emel::error::type err =
      emel::error::cast(emel::speech::encoder::whisper::error::none);
  emel::speech::encoder::whisper::sm machine{};

  emel::speech::encoder::whisper::event::encode bad_rate{
      loaded.contract, pcm, 8000, 1, workspace, output, frames, width, digest};
  bad_rate.error_out = &err;
  CHECK_FALSE(machine.process_event(bad_rate));
  CHECK(err ==
        emel::error::cast(emel::speech::encoder::whisper::error::sample_rate));

  err = emel::error::cast(emel::speech::encoder::whisper::error::none);
  emel::speech::encoder::whisper::event::encode bad_channels{
      loaded.contract, pcm, 16000, 2, workspace, output, frames, width, digest};
  bad_channels.error_out = &err;
  CHECK_FALSE(machine.process_event(bad_channels));
  CHECK(err == emel::error::cast(
                   emel::speech::encoder::whisper::error::channel_count));

  pcm[3] = std::numeric_limits<float>::infinity();
  err = emel::error::cast(emel::speech::encoder::whisper::error::none);
  emel::speech::encoder::whisper::event::encode bad_pcm{
      loaded.contract, pcm, 16000, 1, workspace, output, frames, width, digest};
  bad_pcm.error_out = &err;
  CHECK_FALSE(machine.process_event(bad_pcm));
  CHECK(err ==
        emel::error::cast(emel::speech::encoder::whisper::error::pcm_shape));
}

TEST_CASE("whisper_encoder_rejects_invalid_storage_and_model_contracts") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  const std::vector<float> pcm = deterministic_pcm(320);
  std::vector<float> workspace(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_workspace_floats(
          pcm.size())));
  std::vector<float> output(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_encoder_output_floats(
          pcm.size())));
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t digest = 0;
  emel::error::type err = emel::error::cast(encoder::error::none);

  encoder::detail::execution_contract invalid_contract = loaded.contract;
  invalid_contract.model = nullptr;
  encoder_error_capture error_capture{};
  encoder::sm invalid_model_machine{};
  encoder::event::encode bad_model{
      invalid_contract, pcm,    16000, 1,     workspace,
      output,           frames, width, digest};
  bad_model.on_error =
      emel::callback<void(const encoder::events::encode_error &)>::from<
          encoder_error_capture, record_encoder_error>(&error_capture);
  CHECK_FALSE(invalid_model_machine.process_event(bad_model));
  CHECK(error_capture.calls == 1);
  CHECK(error_capture.err == emel::error::cast(encoder::error::model_invalid));

  std::vector<float> empty_output;
  encoder::sm output_machine{};
  encoder::event::encode bad_output{
      loaded.contract, pcm,    16000, 1,     workspace,
      empty_output,    frames, width, digest};
  bad_output.error_out = &err;
  CHECK_FALSE(output_machine.process_event(bad_output));
  CHECK(err == emel::error::cast(encoder::error::output_capacity));

  std::vector<float> empty_workspace;
  err = emel::error::cast(encoder::error::none);
  encoder::sm workspace_machine{};
  encoder::event::encode bad_workspace{loaded.contract, pcm,    16000,  1,
                                       empty_workspace, output, frames, width,
                                       digest};
  bad_workspace.error_out = &err;
  CHECK_FALSE(workspace_machine.process_event(bad_workspace));
  CHECK(err == emel::error::cast(encoder::error::workspace_capacity));

  auto *variant_tensor = const_cast<emel::model::data::tensor_record *>(
      emel::speech::encoder::whisper::detail::find_tensor(
          *loaded.model, "model.encoder.layers.0.self_attn.k_proj.weight"));
  REQUIRE(variant_tensor != nullptr);
  variant_tensor->type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);

  err = emel::error::cast(encoder::error::none);
  encoder::sm variant_machine{};
  encoder::event::encode bad_variant{
      loaded.contract, pcm, 16000, 1, workspace, output, frames, width, digest};
  bad_variant.error_out = &err;
  CHECK_FALSE(variant_machine.process_event(bad_variant));
  CHECK(err == emel::error::cast(encoder::error::unsupported_variant));
}

TEST_CASE("whisper_encoder_runs_full_q8_encoder_from_public_event") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  const std::vector<float> pcm = deterministic_pcm(320);
  std::vector<float> workspace(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_workspace_floats(
          pcm.size())));
  std::vector<float> output(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_encoder_output_floats(
          pcm.size())));
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t digest = 0;
  emel::error::type err =
      emel::error::cast(emel::speech::encoder::whisper::error::none);
  emel::speech::encoder::whisper::sm machine{};

  emel::speech::encoder::whisper::event::encode request{
      loaded.contract, pcm, 16000, 1, workspace, output, frames, width, digest};
  request.error_out = &err;
  CHECK(machine.process_event(request));
  CHECK(err == emel::error::cast(emel::speech::encoder::whisper::error::none));
  CHECK(frames == 1);
  CHECK(width == emel::speech::encoder::whisper::detail::k_embedding_length);
  CHECK(digest != 0u);
  CHECK(machine.q8_0_dispatch_count() == 1u);
  CHECK(machine.q4_0_dispatch_count() == 0u);
  CHECK(machine.q4_1_dispatch_count() == 0u);

  encoder_done_capture done_capture{};
  std::fill(workspace.begin(), workspace.end(), 0.0f);
  std::fill(output.begin(), output.end(), 0.0f);
  int32_t callback_frames = 0;
  int32_t callback_width = 0;
  uint64_t callback_digest = 0;
  encoder::event::encode callback_request{
      loaded.contract, pcm,    16000,           1,
      workspace,       output, callback_frames, callback_width,
      callback_digest};
  callback_request.on_done =
      emel::callback<void(const encoder::events::encode_done &)>::from<
          encoder_done_capture, record_encoder_done>(&done_capture);
  CHECK(machine.process_event(callback_request));
  CHECK(done_capture.calls == 1);
  CHECK(done_capture.frame_count == callback_frames);
  CHECK(done_capture.width == callback_width);
  CHECK(done_capture.digest == callback_digest);

  std::vector<float> workspace_again = workspace;
  std::fill(output.begin(), output.end(), 0.0f);
  int32_t frames_again = 0;
  int32_t width_again = 0;
  uint64_t digest_again = 0;
  emel::speech::encoder::whisper::event::encode repeat{
      loaded.contract, pcm,          16000,       1,           workspace_again,
      output,          frames_again, width_again, digest_again};
  CHECK(machine.process_event(repeat));
  CHECK(frames_again == frames);
  CHECK(width_again == width);
  CHECK(digest_again == digest);
}

TEST_CASE("whisper_recognizer_runs_fixture_through_public_actor") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  const std::string tokenizer_json = read_text_file(whisper_tokenizer_path());
  const std::vector<float> pcm = deterministic_pcm(320);
  namespace recognizer = emel::speech::recognizer;
  namespace route = emel::speech::recognizer_routes::whisper;

  std::vector<float> encoder_workspace(static_cast<size_t>(
      route::required_encoder_workspace_floats(pcm.size())));
  std::vector<float> encoder_state(
      static_cast<size_t>(route::required_encoder_state_floats(pcm.size())));
  std::vector<float> decoder_workspace(static_cast<size_t>(
      route::required_decoder_workspace_floats(pcm.size())));
  std::vector<float> logits(static_cast<size_t>(route::logits_size()));
  std::array<int32_t, 32> generated_tokens = {};
  std::array<char, 64> transcript = {};
  int32_t transcript_size = -1;
  int32_t selected_token = 0;
  float confidence = 0.0f;
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t encoder_digest = 0u;
  uint64_t decoder_digest = 0u;
  emel::error::type err = emel::error::cast(recognizer::error::none);

  recognizer::sm<route::route> machine{};
  recognizer::event::initialize initialize_ev{
      *loaded.model,
      recognizer::event::tokenizer_assets{
          .model_json = tokenizer_json,
          .sha256 = emel::speech::tokenizer::whisper::tiny_tokenizer_sha256(),
      }};
  initialize_ev.error_out = &err;

  REQUIRE(machine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(recognizer::error::none));

  recognizer::event::recognize recognize_ev{
      *loaded.model,
      recognizer::event::tokenizer_assets{
          .model_json = tokenizer_json,
          .sha256 = emel::speech::tokenizer::whisper::tiny_tokenizer_sha256(),
      },
      std::span<const float>{pcm},
      16000,
      std::span<char>{transcript},
      transcript_size,
      selected_token,
      confidence,
      frames,
      width,
      encoder_digest,
      decoder_digest};
  recognize_ev.storage = recognizer::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  recognize_ev.error_out = &err;

  REQUIRE(machine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(recognizer::error::none));
  CHECK(transcript_size >= 0);
  CHECK(selected_token > 0);
  CHECK(confidence >= 0.0f);
  CHECK(frames == 1);
  CHECK(width == emel::speech::encoder::whisper::detail::k_embedding_length);
  CHECK(encoder_digest != 0u);
  CHECK(decoder_digest != 0u);
}

TEST_CASE("whisper_encoder_routes_q8_linear_f32_aux_variant") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  mark_encoder_aux_tensors_f32(*loaded.model);
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t digest = 0;
  encoder::event::encode request{loaded.contract,
                                 std::span<const float>{},
                                 16000,
                                 1,
                                 std::span<float>{},
                                 std::span<float>{},
                                 frames,
                                 width,
                                 digest};
  encoder::event::encode_ctx run_ctx{};
  const encoder::event::encode_run runtime_ev{request, run_ctx};
  const encoder::action::context action_ctx{};
  CHECK(encoder::guard::guard_model_contract_valid{}(runtime_ev, action_ctx));
  CHECK_FALSE(encoder::guard::guard_q8_0_variant{}(runtime_ev, action_ctx));
  CHECK(encoder::guard::guard_q8_0_f32_aux_variant{}(runtime_ev, action_ctx));
}

TEST_CASE("whisper_encoder_mel_path_consumes_loaded_filter_tensor") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  const std::vector<float> pcm = deterministic_pcm(320);
  std::vector<float> workspace(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_workspace_floats(
          pcm.size())));
  std::vector<float> output(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_encoder_output_floats(
          pcm.size())));
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t original_digest = 0;
  emel::speech::encoder::whisper::sm machine{};
  emel::speech::encoder::whisper::event::encode original{
      loaded.contract, pcm,   16000,          1, workspace, output,
      frames,          width, original_digest};
  REQUIRE(machine.process_event(original));

  auto *mel_filters = const_cast<float *>(static_cast<const float *>(
      emel::speech::encoder::whisper::detail::find_tensor(*loaded.model,
                                                          "mel_filters")
          ->data));
  mel_filters[0] = mel_filters[0] + 0.125f;
  std::fill(workspace.begin(), workspace.end(), 0.0f);
  std::fill(output.begin(), output.end(), 0.0f);
  int32_t mutated_frames = 0;
  int32_t mutated_width = 0;
  uint64_t mutated_digest = 0;
  emel::speech::encoder::whisper::event::encode mutated{
      loaded.contract, pcm,           16000,         1, workspace, output,
      mutated_frames,  mutated_width, mutated_digest};
  REQUIRE(machine.process_event(mutated));
  CHECK(mutated_frames == frames);
  CHECK(mutated_width == width);
  CHECK(mutated_digest != original_digest);
}
