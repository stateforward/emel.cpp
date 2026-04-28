#include "doctest/doctest.h"

#include <algorithm>
#include <array>
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

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/speech/encoder/whisper/detail.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/speech/decoder/whisper/any.hpp"
#include "emel/model/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"
#include "emel/speech/encoder/whisper/sm.hpp"
#include "emel/speech/recognizer_routes/whisper/any.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace {

namespace decoder = emel::speech::decoder::whisper;

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

struct decoder_done_capture {
  int32_t calls = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
};

struct decoder_error_capture {
  int32_t calls = 0;
  emel::error::type err = emel::error::cast(decoder::error::none);
};

void record_decoder_done(decoder_done_capture & capture,
                         const decoder::events::decode_done & done) noexcept {
  ++capture.calls;
  capture.token = done.token;
  capture.confidence = done.confidence;
  capture.digest = done.digest;
}

void record_decoder_error(decoder_error_capture & capture,
                          const decoder::events::decode_error & error) noexcept {
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

std::vector<uint8_t> read_binary_file(const std::filesystem::path & path) {
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

std::string read_text_file(const std::filesystem::path & path) {
  std::ifstream stream(path);
  REQUIRE(stream.good());
  return std::string{std::istreambuf_iterator<char>{stream},
                     std::istreambuf_iterator<char>{}};
}

void materialize_tensor_names_from_file(emel::model::data & model,
                                        const std::vector<uint8_t> & file_bytes) {
  model.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    auto & tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= file_bytes.size());
    REQUIRE(static_cast<size_t>(model.name_bytes_used) + length <= model.name_storage.size());
    std::memcpy(model.name_storage.data() + model.name_bytes_used,
                file_bytes.data() + source_offset,
                length);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
}

bool load_whisper_fixture_binding(const std::vector<uint8_t> & file_bytes,
                                  std::vector<uint8_t> & kv_arena,
                                  std::vector<emel::gguf::loader::kv_entry> & kv_entries,
                                  emel::model::data & model_out) {
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
      std::span<const uint8_t>{file_bytes}, requirements, on_probe_done, on_probe_error};
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
      on_bind_done,
      on_bind_error};
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
  emel::speech::encoder::whisper::detail::execution_contract encoder_contract = {};
  decoder::detail::execution_contract decoder_contract = {};
};

void mark_decoder_aux_tensors_f32(emel::model::data & model) {
  for (uint32_t index = 0; index < model.n_tensors; ++index) {
    auto & tensor = model.tensors[index];
    const auto name = emel::model::tensor_name_view(model, tensor);
    if (name.starts_with("model.decoder.") &&
        (tensor.n_dims == 1 || name == "model.decoder.embed_positions.weight")) {
      tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
    }
  }
}

loaded_whisper_fixture load_fixture_or_skip() {
  const auto fixture_path = whisper_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping Whisper decoder fixture test because fixture is missing: "
            << fixture_path.string());
    return {};
  }

  loaded_whisper_fixture loaded{};
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(fixture_path);
  REQUIRE(load_whisper_fixture_binding(loaded.file_bytes,
                                       loaded.kv_arena,
                                       loaded.kv_entries,
                                       *loaded.model));
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{loaded.kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{loaded.kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *loaded.model));
  loaded.model->weights_data = loaded.file_bytes.data();
  loaded.model->weights_size = loaded.file_bytes.size();
  loaded.model->weights_mapped = true;
  materialize_tensor_names_from_file(*loaded.model, loaded.file_bytes);
  emel::model::whisper::detail::execution_contract model_contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(*loaded.model,
                                                                 model_contract) ==
          emel::error::cast(emel::model::loader::error::none));
  loaded.encoder_contract =
      emel::speech::encoder::whisper::detail::bind_execution_contract(*loaded.model);
  loaded.decoder_contract = decoder::detail::bind_execution_contract(*loaded.model);
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

struct encoded_fixture {
  std::vector<float> encoder_workspace = {};
  std::vector<float> encoder_state = {};
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t digest = 0;
};

encoded_fixture encode_fixture_audio(const loaded_whisper_fixture & loaded) {
  const std::vector<float> pcm = deterministic_pcm(320);
  encoded_fixture encoded{};
  encoded.encoder_workspace.resize(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_workspace_floats(pcm.size())));
  encoded.encoder_state.resize(static_cast<size_t>(
      emel::speech::encoder::whisper::detail::required_encoder_output_floats(pcm.size())));
  emel::speech::encoder::whisper::sm encoder{};
  emel::speech::encoder::whisper::event::encode request{loaded.encoder_contract,
                                                pcm,
                                                16000,
                                                1,
                                                encoded.encoder_workspace,
                                                encoded.encoder_state,
                                                encoded.frames,
                                                encoded.width,
                                                encoded.digest};
  REQUIRE(encoder.process_event(request));
  REQUIRE(encoded.frames > 0);
  REQUIRE(encoded.width == emel::speech::encoder::whisper::detail::k_embedding_length);
  return encoded;
}

struct zero_q4_decoder_fixture {
  std::unique_ptr<emel::model::data> model = {};
  std::vector<emel::kernel::detail::quant::block_q4_0> q4_blocks = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> q8_blocks = {};
};

void add_test_tensor(emel::model::data &model, const std::string_view name,
                     const int32_t type, const int32_t n_dims,
                     const std::array<int64_t, 4> dims, const void *data,
                     const uint64_t data_size) {
  REQUIRE(model.n_tensors < model.tensors.size());
  REQUIRE(static_cast<size_t>(model.name_bytes_used) + name.size() <=
          model.name_storage.size());
  auto &tensor = model.tensors[model.n_tensors++];
  tensor.name_offset = model.name_bytes_used;
  tensor.name_length = static_cast<uint32_t>(name.size());
  std::memcpy(model.name_storage.data() + model.name_bytes_used, name.data(),
              name.size());
  model.name_bytes_used += static_cast<uint32_t>(name.size());
  tensor.type = type;
  tensor.n_dims = n_dims;
  tensor.dims = dims;
  tensor.data = data;
  tensor.data_size = data_size;
}

zero_q4_decoder_fixture make_zero_q4_decoder_fixture() {
  namespace quant = emel::kernel::detail::quant;
  namespace whisper = emel::speech::decoder::whisper::detail;

  zero_q4_decoder_fixture fixture{};
  fixture.model = std::make_unique<emel::model::data>();
  fixture.q4_blocks.resize(static_cast<size_t>(
      static_cast<uint64_t>(whisper::k_vocab_size) *
      (static_cast<uint64_t>(whisper::k_embedding_length) / quant::QK4_0)));
  fixture.q8_blocks.resize(static_cast<size_t>(
      static_cast<uint64_t>(whisper::k_decoder_sequence_token_count) *
      (static_cast<uint64_t>(whisper::k_embedding_length) / quant::QK8_0)));
  for (auto &block : fixture.q4_blocks) {
    block.d = quant::fp32_to_fp16(0.0f);
    block.qs.fill(0x88u);
  }
  for (auto &block : fixture.q8_blocks) {
    block.d = quant::fp32_to_fp16(0.0f);
    block.qs.fill(0);
  }

  auto add_q4_weight = [&](const std::string &name, const int64_t in,
                           const int64_t out) {
    add_test_tensor(*fixture.model, name,
                    static_cast<int32_t>(emel::kernel::detail::dtype_q4_0), 2,
                    {in, out, 0, 0}, fixture.q4_blocks.data(),
                    fixture.q4_blocks.size() * sizeof(quant::block_q4_0));
  };
  auto add_q8_aux = [&](const std::string &name, const int32_t n_dims,
                        const std::array<int64_t, 4> dims) {
    add_test_tensor(*fixture.model, name,
                    static_cast<int32_t>(emel::kernel::detail::dtype_q8_0),
                    n_dims, dims, fixture.q8_blocks.data(),
                    fixture.q8_blocks.size() * sizeof(quant::block_q8_0));
  };

  add_q4_weight("model.decoder.embed_tokens.weight",
                whisper::k_embedding_length, whisper::k_vocab_size);
  add_q8_aux("model.decoder.embed_positions.weight", 2,
             {whisper::k_embedding_length, whisper::k_decoder_sequence_token_count,
              0, 0});
  add_q8_aux("model.decoder.layer_norm.weight", 1,
             {whisper::k_embedding_length, 0, 0, 0});
  add_q8_aux("model.decoder.layer_norm.bias", 1,
             {whisper::k_embedding_length, 0, 0, 0});

  for (int32_t layer = 0; layer < whisper::k_decoder_block_count; ++layer) {
    const std::string prefix =
        "model.decoder.layers." + std::to_string(layer) + ".";
    add_q8_aux(prefix + "self_attn_layer_norm.weight", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q8_aux(prefix + "self_attn_layer_norm.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "self_attn.q_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q8_aux(prefix + "self_attn.q_proj.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "self_attn.k_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q4_weight(prefix + "self_attn.v_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q8_aux(prefix + "self_attn.v_proj.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "self_attn.out_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q8_aux(prefix + "self_attn.out_proj.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q8_aux(prefix + "encoder_attn_layer_norm.weight", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q8_aux(prefix + "encoder_attn_layer_norm.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "encoder_attn.q_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q8_aux(prefix + "encoder_attn.q_proj.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "encoder_attn.out_proj.weight",
                  whisper::k_embedding_length, whisper::k_embedding_length);
    add_q8_aux(prefix + "encoder_attn.out_proj.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q8_aux(prefix + "final_layer_norm.weight", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q8_aux(prefix + "final_layer_norm.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
    add_q4_weight(prefix + "fc1.weight", whisper::k_embedding_length,
                  whisper::k_feed_forward_length);
    add_q8_aux(prefix + "fc1.bias", 1,
               {whisper::k_feed_forward_length, 0, 0, 0});
    add_q4_weight(prefix + "fc2.weight", whisper::k_feed_forward_length,
                  whisper::k_embedding_length);
    add_q8_aux(prefix + "fc2.bias", 1,
               {whisper::k_embedding_length, 0, 0, 0});
  }

  return fixture;
}

template <uint64_t In, uint64_t Out>
void exercise_q4_decoder_linear_shape() {
  namespace quant = emel::kernel::detail::quant;
  namespace whisper = emel::speech::decoder::whisper::detail;

  constexpr uint64_t q4_block_count = In / quant::QK4_0;
  std::vector<quant::block_q4_0> q4_0_blocks(
      static_cast<size_t>(Out * q4_block_count));
  std::vector<quant::block_q4_1> q4_1_blocks(
      static_cast<size_t>(Out * q4_block_count));
  for (auto &block : q4_0_blocks) {
    block.d = quant::fp32_to_fp16(0.0f);
    block.qs.fill(0x88u);
  }
  for (auto &block : q4_1_blocks) {
    block.d = quant::fp32_to_fp16(0.0f);
    block.m = quant::fp32_to_fp16(0.0f);
    block.qs.fill(0x00u);
  }

  emel::model::data::tensor_record q4_0_tensor{};
  q4_0_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_q4_0);
  q4_0_tensor.n_dims = 2;
  q4_0_tensor.dims = {
      static_cast<int64_t>(In), static_cast<int64_t>(Out), 0, 0};
  q4_0_tensor.data = q4_0_blocks.data();
  q4_0_tensor.data_size = q4_0_blocks.size() * sizeof(quant::block_q4_0);

  emel::model::data::tensor_record q4_1_tensor{};
  q4_1_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_q4_1);
  q4_1_tensor.n_dims = 2;
  q4_1_tensor.dims = {
      static_cast<int64_t>(In), static_cast<int64_t>(Out), 0, 0};
  q4_1_tensor.data = q4_1_blocks.data();
  q4_1_tensor.data_size = q4_1_blocks.size() * sizeof(quant::block_q4_1);

  constexpr uint64_t q8_bias_block_count = Out / quant::QK8_0;
  std::vector<quant::block_q8_0> bias_blocks(
      static_cast<size_t>(q8_bias_block_count));
  for (auto &block : bias_blocks) {
    block.d = quant::fp32_to_fp16(0.0f);
    block.qs.fill(0);
  }
  emel::model::data::tensor_record bias_tensor{};
  bias_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_q8_0);
  bias_tensor.n_dims = 1;
  bias_tensor.dims = {static_cast<int64_t>(Out), 0, 0, 0};
  bias_tensor.data = bias_blocks.data();
  bias_tensor.data_size = bias_blocks.size() * sizeof(quant::block_q8_0);

  std::vector<float> input(static_cast<size_t>(In), 1.0f);
  std::vector<float> output(static_cast<size_t>(Out), -1.0f);
  whisper::linear<whisper::linear_weight_variant::q4_0, In, Out>(
      q4_0_tensor, bias_tensor, input.data(), output.data());
  CHECK(output[0] == doctest::Approx(0.0f));

  std::fill(output.begin(), output.end(), -1.0f);
  whisper::linear<whisper::linear_weight_variant::q4_1, In, Out>(
      q4_1_tensor, bias_tensor, input.data(), output.data());
  CHECK(output[0] == doctest::Approx(0.0f));

  if constexpr (In == 384u && Out == 384u) {
    std::fill(output.begin(), output.end(), -1.0f);
    whisper::linear_no_bias<whisper::linear_weight_variant::q4_0, In, Out>(
        q4_0_tensor, input.data(), output.data());
    CHECK(output[0] == doctest::Approx(0.0f));

    std::fill(output.begin(), output.end(), -1.0f);
    whisper::linear_no_bias<whisper::linear_weight_variant::q4_1, In, Out>(
        q4_1_tensor, input.data(), output.data());
    CHECK(output[0] == doctest::Approx(0.0f));
  }
}

}  // namespace

TEST_CASE("whisper_decoder_runtime_owns_decode_detail_dependencies") {
  const auto root = repo_root();
  const std::array<std::filesystem::path, 3> production_files{
      root / "src" / "emel" / "speech" / "decoder" / "whisper" / "actions.hpp",
      root / "src" / "emel" / "speech" / "decoder" / "whisper" / "guards.hpp",
      root / "src" / "emel" / "speech" / "decoder" / "whisper" / "detail.hpp",
  };

  for (const auto &path : production_files) {
    const std::string source = read_text_file(path);
    CHECK(source.find("emel/speech/encoder/whisper/detail.hpp") == std::string::npos);
    CHECK(source.find("encoder::whisper::detail") == std::string::npos);
  }

  const std::string decoder_detail = read_text_file(production_files[2]);
  CHECK(decoder_detail.find("run_decoder_sequence") != std::string::npos);
  CHECK(decoder_detail.find("select_greedy_timestamp_aware_token") != std::string::npos);
}

TEST_CASE("whisper_decoder_detail_timestamp_blocking_is_decoder_owned") {
  namespace whisper = emel::speech::decoder::whisper::detail;

  std::vector<float> logits(static_cast<size_t>(whisper::k_vocab_size),
                            -1000.0f);
  logits[42] = 100.0f;
  logits[static_cast<size_t>(whisper::k_token_timestamp_begin)] = 0.0f;
  const std::array<int32_t, 2> generated{42,
                                         whisper::k_token_timestamp_begin};
  const whisper::decode_policy_runtime policy{};
  float confidence = 0.0f;
  const int32_t token = whisper::select_greedy_timestamp_aware_token(
      policy, logits.data(), generated.data(), generated.size(), false,
      confidence);
  CHECK(token >= whisper::k_token_timestamp_begin);
}

TEST_CASE("whisper_decoder_detail_timestamp_policy_suppresses_control_tokens") {
  namespace whisper = emel::speech::decoder::whisper::detail;

  const whisper::decode_policy_runtime policy{};

  std::vector<float> initial_logits(
      static_cast<size_t>(whisper::k_vocab_size), -1000.0f);
  initial_logits[static_cast<size_t>(policy.eot)] = 500.0f;
  initial_logits[static_cast<size_t>(policy.space)] = 400.0f;
  initial_logits[42] = 10.0f;
  float initial_confidence = 0.0f;
  const int32_t initial_token = whisper::select_greedy_timestamp_aware_token(
      policy, initial_logits.data(), nullptr, 0u, true, initial_confidence);
  CHECK(initial_token == 42);
  CHECK(initial_confidence == doctest::Approx(10.0f));

  std::vector<float> control_logits(
      static_cast<size_t>(whisper::k_vocab_size), -1000.0f);
  control_logits[static_cast<size_t>(policy.sot)] = 600.0f;
  control_logits[static_cast<size_t>(policy.translate)] = 500.0f;
  control_logits[static_cast<size_t>(policy.transcribe)] = 400.0f;
  control_logits[static_cast<size_t>(policy.no_speech)] = 300.0f;
  control_logits[static_cast<size_t>(policy.notimestamps)] = 200.0f;
  control_logits[77] = 20.0f;
  float control_confidence = 0.0f;
  const int32_t control_token = whisper::select_greedy_timestamp_aware_token(
      policy, control_logits.data(), nullptr, 0u, false, control_confidence);
  CHECK(control_token == 77);
  CHECK(control_confidence == doctest::Approx(20.0f));
}

TEST_CASE("whisper_decoder_generation_budget_matches_text_context") {
  namespace whisper = emel::speech::decoder::whisper::detail;

  CHECK(whisper::k_decoder_sequence_token_count == 448);
  CHECK(whisper::k_max_generated_token_count ==
        whisper::k_decoder_sequence_token_count -
            whisper::k_decoder_prompt_token_count);
}

TEST_CASE("whisper_decoder_public_generation_budget_hooks_match_detail") {
  namespace route = emel::speech::recognizer_routes::whisper;
  namespace whisper = emel::speech::decoder::whisper::detail;

  CHECK(decoder::max_generated_token_count() ==
        whisper::k_max_generated_token_count);
  CHECK(route::max_generated_token_count() ==
        whisper::k_max_generated_token_count);
}

TEST_CASE("whisper_decoder_detail_reads_f32_aux_and_q4_linear_rows") {
  namespace quant = emel::kernel::detail::quant;
  namespace whisper = emel::speech::decoder::whisper::detail;

  auto model = std::make_unique<emel::model::data>();
  CHECK(whisper::find_tensor(*model, "missing.tensor") == nullptr);
  CHECK(whisper::required_decoder_workspace_floats(1024u) >
        whisper::required_decoder_workspace_floats(1u));

  emel::model::data::tensor_record invalid_shape{};
  CHECK_FALSE(whisper::tensor_has_shape(invalid_shape, 1, {2, 0, 0, 0}));

  const std::array<float, 2> aux_values{1.25f, -2.5f};
  emel::model::data::tensor_record aux_tensor{};
  aux_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
  aux_tensor.n_dims = 1;
  aux_tensor.dims = {2, 1, 1, 1};
  aux_tensor.data = aux_values.data();
  aux_tensor.data_size = sizeof(float) * aux_values.size();
  CHECK(whisper::read_aux_vector<whisper::aux_weight_variant::f32>(
            aux_tensor, 0u) == doctest::Approx(1.25f));
  CHECK(whisper::read_aux_matrix<whisper::aux_weight_variant::f32>(
            aux_tensor, 1u) == doctest::Approx(-2.5f));

  std::array<quant::block_q4_0, 1> q4_0_blocks{};
  q4_0_blocks[0].d = quant::fp32_to_fp16(0.5f);
  q4_0_blocks[0].qs.fill(0x98u);
  emel::model::data::tensor_record q4_0_tensor{};
  q4_0_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_q4_0);
  q4_0_tensor.n_dims = 2;
  q4_0_tensor.dims = {static_cast<int64_t>(quant::QK4_0), 1, 1, 1};
  q4_0_tensor.data = q4_0_blocks.data();
  q4_0_tensor.data_size = sizeof(q4_0_blocks);

  std::array<quant::block_q4_1, 1> q4_1_blocks{};
  q4_1_blocks[0].d = quant::fp32_to_fp16(0.25f);
  q4_1_blocks[0].m = quant::fp32_to_fp16(1.5f);
  q4_1_blocks[0].qs.fill(0x42u);
  emel::model::data::tensor_record q4_1_tensor{};
  q4_1_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_q4_1);
  q4_1_tensor.n_dims = 2;
  q4_1_tensor.dims = {static_cast<int64_t>(quant::QK4_1), 1, 1, 1};
  q4_1_tensor.data = q4_1_blocks.data();
  q4_1_tensor.data_size = sizeof(q4_1_blocks);

  std::array<float, static_cast<size_t>(quant::QK4_0)> input{};
  input.fill(1.0f);
  std::array<float, 1> output{};

  CHECK(whisper::read_matrix_q4_0_value(q4_0_tensor, 0u, 0u) ==
        doctest::Approx(0.0f));
  CHECK(whisper::read_matrix_q4_0_value(q4_0_tensor, 0u, 16u) ==
        doctest::Approx(0.5f));
  CHECK(whisper::dot_linear_row<whisper::linear_weight_variant::q4_0>(
            q4_0_tensor, 0u, input.data(), input.size()) ==
        doctest::Approx(8.0f));
  whisper::linear_no_bias<whisper::linear_weight_variant::q4_0,
                          quant::QK4_0, 1u>(
      q4_0_tensor, input.data(), output.data());
  CHECK(output[0] == doctest::Approx(8.0f));

  CHECK(whisper::read_matrix_q4_1_value(q4_1_tensor, 0u, 0u) ==
        doctest::Approx(2.0f));
  CHECK(whisper::read_matrix_q4_1_value(q4_1_tensor, 0u, 16u) ==
        doctest::Approx(2.5f));
  CHECK(whisper::dot_linear_row<whisper::linear_weight_variant::q4_1>(
            q4_1_tensor, 0u, input.data(), input.size()) ==
        doctest::Approx(72.0f));
  whisper::linear<whisper::linear_weight_variant::q4_1, quant::QK4_1, 1u,
                  whisper::aux_weight_variant::f32>(
      q4_1_tensor, aux_tensor, input.data(), output.data());
  CHECK(output[0] == doctest::Approx(73.25f));

  std::array<float, static_cast<size_t>(whisper::k_embedding_length)> norm_in{};
  std::array<float, static_cast<size_t>(whisper::k_embedding_length)> norm_w{};
  std::array<float, static_cast<size_t>(whisper::k_embedding_length)> norm_b{};
  std::array<float, static_cast<size_t>(whisper::k_embedding_length)> norm_out{};
  for (size_t index = 0; index < norm_in.size(); ++index) {
    norm_in[index] = static_cast<float>(index % 7u);
    norm_w[index] = 1.0f;
  }
  emel::model::data::tensor_record norm_w_tensor{};
  norm_w_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
  norm_w_tensor.n_dims = 1;
  norm_w_tensor.dims = {whisper::k_embedding_length, 0, 0, 0};
  norm_w_tensor.data = norm_w.data();
  norm_w_tensor.data_size = sizeof(float) * norm_w.size();
  emel::model::data::tensor_record norm_b_tensor = norm_w_tensor;
  norm_b_tensor.data = norm_b.data();
  whisper::layer_norm_frame<whisper::aux_weight_variant::f32>(
      norm_in.data(), norm_w_tensor, norm_b_tensor, norm_out.data());
  CHECK(std::isfinite(norm_out[0]));

  std::array<float, 3> softmax_values{3.0f, 2.0f, 1.0f};
  whisper::softmax(softmax_values.data(), softmax_values.size());
  CHECK(softmax_values[0] > softmax_values[1]);
}

TEST_CASE("whisper_decoder_detail_exercises_compiled_q4_linear_shapes") {
  exercise_q4_decoder_linear_shape<384u, 384u>();
  exercise_q4_decoder_linear_shape<384u, 1536u>();
  exercise_q4_decoder_linear_shape<1536u, 384u>();
}

TEST_CASE("whisper_decoder_detail_exercises_q4_logits_path") {
  namespace whisper = emel::speech::decoder::whisper::detail;

  auto fixture = make_zero_q4_decoder_fixture();
  constexpr uint64_t encoder_frames = 1u;
  std::array<float, static_cast<size_t>(whisper::k_decoder_block_count *
                                        whisper::k_embedding_length)>
      cross_k = {};
  std::array<float, static_cast<size_t>(whisper::k_decoder_block_count *
                                        whisper::k_embedding_length)>
      cross_v = {};
  const std::array<int32_t, 1> tokens{0};
  std::vector<float> workspace(static_cast<size_t>(
      whisper::required_decoder_workspace_floats(encoder_frames)));
  std::vector<float> logits(static_cast<size_t>(whisper::k_vocab_size), -1.0f);
  float confidence = -1.0f;
  uint64_t digest = 0u;

  whisper::compute_decoder_logits_for_tokens<
      whisper::linear_weight_variant::q4_0>(
      *fixture.model, encoder_frames, cross_k.data(), cross_v.data(),
      tokens.data(), tokens.size(), workspace.data(), logits.data(),
      confidence, digest);

  CHECK(confidence == doctest::Approx(0.0f));
  CHECK(logits[0] == doctest::Approx(0.0f));
  CHECK(digest != 0u);
}

TEST_CASE("whisper_decoder_rejects_invalid_runtime_capacity") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }
  auto encoded = encode_fixture_audio(loaded);

  std::vector<float> workspace(static_cast<size_t>(
      decoder::detail::required_decoder_workspace_floats(
          static_cast<uint64_t>(encoded.frames))));
  std::vector<float> logits(static_cast<size_t>(decoder::detail::k_vocab_size));
  const auto &policy =
      emel::speech::tokenizer::whisper::tiny_asr_decode_policy();
  auto unsupported_policy = policy;
  unsupported_policy.suppress_translate = false;
  std::array<int32_t, 4> generated_tokens = {};
  int32_t generated_token_count = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
  emel::error::type err = emel::error::cast(emel::speech::decoder::whisper::error::none);
  emel::speech::decoder::whisper::sm decoder{};

  emel::speech::decoder::whisper::event::decode bad_policy{loaded.decoder_contract,
                                                  encoded.encoder_state,
                                                  encoded.frames,
                                                  unsupported_policy,
                                                  generated_tokens,
                                                  generated_token_count,
                                                  workspace,
                                                  logits,
                                                  token,
                                                  confidence,
                                                  digest};
  bad_policy.error_out = &err;
  CHECK_FALSE(decoder.process_event(bad_policy));
  CHECK(err == emel::error::cast(emel::speech::decoder::whisper::error::decode_policy));

  std::vector<float> small_logits(16u);
  err = emel::error::cast(emel::speech::decoder::whisper::error::none);
  emel::speech::decoder::whisper::event::decode bad_logits{loaded.decoder_contract,
                                                   encoded.encoder_state,
                                                   encoded.frames,
                                                   policy,
                                                   generated_tokens,
                                                   generated_token_count,
                                                   workspace,
                                                   small_logits,
                                                   token,
                                                   confidence,
                                                   digest};
  bad_logits.error_out = &err;
  CHECK_FALSE(decoder.process_event(bad_logits));
  CHECK(err == emel::error::cast(emel::speech::decoder::whisper::error::logits_capacity));

  std::vector<int32_t> empty_generated_tokens;
  err = emel::error::cast(emel::speech::decoder::whisper::error::none);
  emel::speech::decoder::whisper::event::decode bad_generated_capacity{
      loaded.decoder_contract,
      encoded.encoder_state,
      encoded.frames,
      policy,
      empty_generated_tokens,
      generated_token_count,
      workspace,
      logits,
      token,
      confidence,
      digest};
  bad_generated_capacity.error_out = &err;
  CHECK_FALSE(decoder.process_event(bad_generated_capacity));
  CHECK(err == emel::error::cast(
                   emel::speech::decoder::whisper::error::generated_token_capacity));
}

TEST_CASE("whisper_decoder_rejects_invalid_state_storage_and_model_contracts") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }
  auto encoded = encode_fixture_audio(loaded);

  std::vector<float> workspace(static_cast<size_t>(
      decoder::detail::required_decoder_workspace_floats(
          static_cast<uint64_t>(encoded.frames))));
  std::vector<float> logits(static_cast<size_t>(decoder::detail::k_vocab_size));
  const auto &policy =
      emel::speech::tokenizer::whisper::tiny_asr_decode_policy();
  std::array<int32_t, 4> generated_tokens = {};
  int32_t generated_token_count = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
  emel::error::type err = emel::error::cast(decoder::error::none);

  decoder::detail::execution_contract invalid_contract = loaded.decoder_contract;
  invalid_contract.model = nullptr;
  decoder_error_capture error_capture{};
  decoder::sm invalid_model_machine{};
  decoder::event::decode bad_model{invalid_contract,
                                   encoded.encoder_state,
                                   encoded.frames,
                                   policy,
                                   generated_tokens,
                                   generated_token_count,
                                   workspace,
                                   logits,
                                   token,
                                   confidence,
                                   digest};
  bad_model.on_error =
      emel::callback<void(const decoder::events::decode_error &)>::from<
          decoder_error_capture, record_decoder_error>(&error_capture);
  CHECK_FALSE(invalid_model_machine.process_event(bad_model));
  CHECK(error_capture.calls == 1);
  CHECK(error_capture.err == emel::error::cast(decoder::error::model_invalid));

  std::vector<float> empty_encoder_state;
  decoder::sm encoder_state_machine{};
  decoder::event::decode bad_encoder_state{loaded.decoder_contract,
                                           empty_encoder_state,
                                           encoded.frames,
                                           policy,
                                           generated_tokens,
                                           generated_token_count,
                                           workspace,
                                           logits,
                                           token,
                                           confidence,
                                           digest};
  bad_encoder_state.error_out = &err;
  CHECK_FALSE(encoder_state_machine.process_event(bad_encoder_state));
  CHECK(err == emel::error::cast(decoder::error::encoder_state));

  std::vector<float> empty_workspace;
  err = emel::error::cast(decoder::error::none);
  decoder::sm workspace_machine{};
  decoder::event::decode bad_workspace{loaded.decoder_contract,
                                       encoded.encoder_state,
                                       encoded.frames,
                                       policy,
                                       generated_tokens,
                                       generated_token_count,
                                       empty_workspace,
                                       logits,
                                       token,
                                       confidence,
                                       digest};
  bad_workspace.error_out = &err;
  CHECK_FALSE(workspace_machine.process_event(bad_workspace));
  CHECK(err == emel::error::cast(decoder::error::workspace_capacity));

  auto * variant_tensor = const_cast<emel::model::data::tensor_record *>(
      emel::speech::encoder::whisper::detail::find_tensor(
          *loaded.model, "model.decoder.layers.0.self_attn.k_proj.weight"));
  REQUIRE(variant_tensor != nullptr);
  variant_tensor->type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);

  err = emel::error::cast(decoder::error::none);
  decoder::sm variant_machine{};
  decoder::event::decode bad_variant{loaded.decoder_contract,
                                     encoded.encoder_state,
                                     encoded.frames,
                                     policy,
                                     generated_tokens,
                                     generated_token_count,
                                     workspace,
                                     logits,
                                     token,
                                     confidence,
                                     digest};
  bad_variant.error_out = &err;
  CHECK_FALSE(variant_machine.process_event(bad_variant));
  CHECK(err == emel::error::cast(decoder::error::unsupported_variant));
}

TEST_CASE("whisper_decoder_runs_first_q8_token_from_public_event") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }
  auto encoded = encode_fixture_audio(loaded);

  std::vector<float> workspace(static_cast<size_t>(
      decoder::detail::required_decoder_workspace_floats(
          static_cast<uint64_t>(encoded.frames))));
  std::vector<float> logits(static_cast<size_t>(decoder::detail::k_vocab_size));
  const auto &policy =
      emel::speech::tokenizer::whisper::tiny_asr_decode_policy();
  std::array<int32_t, 4> generated_tokens = {};
  int32_t generated_token_count = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
  emel::error::type err = emel::error::cast(emel::speech::decoder::whisper::error::none);
  emel::speech::decoder::whisper::sm decoder{};

  emel::speech::decoder::whisper::event::decode request{loaded.decoder_contract,
                                                encoded.encoder_state,
                                                encoded.frames,
                                                policy,
                                                generated_tokens,
                                                generated_token_count,
                                                workspace,
                                                logits,
                                                token,
                                                confidence,
                                                digest};
  request.error_out = &err;
  CHECK(decoder.process_event(request));
  CHECK(err == emel::error::cast(emel::speech::decoder::whisper::error::none));
  CHECK(token >= 0);
  CHECK(token < decoder::detail::k_vocab_size);
  CHECK(std::isfinite(confidence));
  CHECK(logits[static_cast<size_t>(token)] == doctest::Approx(confidence));
  CHECK(generated_token_count > 0);
  CHECK(generated_tokens[0] == token);
  CHECK(digest != 0u);
  CHECK(decoder.q8_0_dispatch_count() == 1u);
  CHECK(decoder.q4_0_dispatch_count() == 0u);
  CHECK(decoder.q4_1_dispatch_count() == 0u);

  decoder_done_capture done_capture{};
  std::fill(workspace.begin(), workspace.end(), 0.0f);
  std::fill(logits.begin(), logits.end(), 0.0f);
  std::array<int32_t, 4> callback_generated_tokens = {};
  int32_t callback_generated_token_count = 0;
  int32_t callback_token = 0;
  float callback_confidence = 0.0f;
  uint64_t callback_digest = 0;
  decoder::event::decode callback_request{loaded.decoder_contract,
                                          encoded.encoder_state,
                                          encoded.frames,
                                          policy,
                                          callback_generated_tokens,
                                          callback_generated_token_count,
                                          workspace,
                                          logits,
                                          callback_token,
                                          callback_confidence,
                                          callback_digest};
  callback_request.on_done =
      emel::callback<void(const decoder::events::decode_done &)>::from<
          decoder_done_capture, record_decoder_done>(&done_capture);
  CHECK(decoder.process_event(callback_request));
  CHECK(done_capture.calls == 1);
  CHECK(done_capture.token == callback_token);
  CHECK(done_capture.confidence == doctest::Approx(callback_confidence));
  CHECK(done_capture.digest == callback_digest);

  std::vector<float> workspace_again = workspace;
  std::vector<float> logits_again(logits.size());
  std::array<int32_t, 4> generated_tokens_again = {};
  int32_t generated_token_count_again = 0;
  int32_t token_again = 0;
  float confidence_again = 0.0f;
  uint64_t digest_again = 0;
  emel::speech::decoder::whisper::event::decode repeat{loaded.decoder_contract,
                                               encoded.encoder_state,
                                               encoded.frames,
                                               policy,
                                               generated_tokens_again,
                                               generated_token_count_again,
                                               workspace_again,
                                               logits_again,
                                               token_again,
                                               confidence_again,
                                               digest_again};
  CHECK(decoder.process_event(repeat));
  CHECK(token_again == token);
  CHECK(confidence_again == doctest::Approx(confidence));
  CHECK(generated_token_count_again == generated_token_count);
  CHECK(generated_tokens_again[0] == generated_tokens[0]);
  CHECK(digest_again == digest);

  std::fill(workspace.begin(), workspace.end(), 0.0f);
  std::fill(logits.begin(), logits.end(), 0.0f);
  std::vector<int32_t> stop_generated_tokens(
      static_cast<size_t>(decoder::max_generated_token_count()) + 16u);
  uint64_t stop_generated_token_count = 0u;
  int32_t stop_token = 0;
  float stop_confidence = 0.0f;
  decoder::detail::decode_policy_runtime stop_policy{};
  stop_policy.eot = -1;
  stop_policy.sot = -2;
  stop_policy.translate = -3;
  stop_policy.transcribe = -4;
  stop_policy.no_speech = -5;
  stop_policy.notimestamps = -6;
  stop_policy.timestamp_begin = 0;
  stop_policy.space = -7;
  const uint64_t stop_digest =
      decoder::detail::run_decoder_sequence<
          decoder::detail::linear_weight_variant::q8_0>(
          *loaded.decoder_contract.model, encoded.encoder_state.data(),
          static_cast<uint64_t>(encoded.frames), stop_policy,
          policy.prompt_tokens.data(), policy.prompt_tokens.size(),
          workspace.data(), logits.data(), stop_generated_tokens.data(),
          stop_generated_tokens.size(), stop_generated_token_count, stop_token,
          stop_confidence);
  CHECK(stop_generated_token_count == 2u);
  CHECK(stop_generated_tokens[1] >= stop_policy.timestamp_begin);
  CHECK(stop_digest != 0u);
}

TEST_CASE("whisper_decoder_routes_q8_linear_f32_aux_variant") {
  auto loaded = load_fixture_or_skip();
  if (loaded.model == nullptr) {
    return;
  }

  mark_decoder_aux_tensors_f32(*loaded.model);
  std::vector<float> encoder_state(static_cast<size_t>(
      decoder::detail::k_embedding_length));
  std::vector<float> workspace;
  std::vector<float> logits;
  const auto &policy =
      emel::speech::tokenizer::whisper::tiny_asr_decode_policy();
  std::array<int32_t, 4> generated_tokens = {};
  int32_t generated_token_count = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  uint64_t digest = 0;
  decoder::event::decode request{loaded.decoder_contract,
                                 encoder_state,
                                 1,
                                 policy,
                                 generated_tokens,
                                 generated_token_count,
                                 workspace,
                                 logits,
                                 token,
                                 confidence,
                                 digest};
  decoder::event::decode_ctx run_ctx{};
  const decoder::event::decode_run runtime_ev{request, run_ctx};
  const decoder::action::context action_ctx{};
  CHECK(decoder::guard::guard_model_contract_valid{}(runtime_ev, action_ctx));
  CHECK_FALSE(decoder::guard::guard_q8_0_variant{}(runtime_ev, action_ctx));
  CHECK(decoder::guard::guard_q8_0_f32_aux_variant{}(runtime_ev, action_ctx));
}
