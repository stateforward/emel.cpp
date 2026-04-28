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
#include "emel/model/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"
#include "emel/speech/encoder/whisper/sm.hpp"
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
