#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace emel::tests::embeddings::te_fixture {

namespace embedding_detail = emel::embeddings::generator::detail;

struct loaded_te_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = std::make_unique<emel::model::data>();
};

inline std::filesystem::path repo_root() {
  return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path();
}

inline std::filesystem::path te_fixture_path() {
  return repo_root() / "tests" / "models" / "TE-75M-q8_0.gguf";
}

inline std::filesystem::path te_vocab_path() {
  return repo_root() / "tests" / "models" / "mdbr-leaf-ir-vocab.txt";
}

inline std::filesystem::path te_prompt_path(const std::string_view name) {
  return repo_root() / "tests" / "embeddings" / "fixtures" / "te75m" /
      std::string{name};
}

inline bool te_assets_present() {
  return std::filesystem::exists(te_fixture_path()) &&
      std::filesystem::exists(te_vocab_path());
}

inline std::vector<uint8_t> read_binary_file(const std::filesystem::path & path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE_MESSAGE(stream.good(), "failed to open binary fixture: " << path.string());

  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  REQUIRE(size > 0);
  stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(stream.good());
  return bytes;
}

inline std::string read_text_file(const std::filesystem::path & path) {
  std::ifstream stream(path);
  REQUIRE_MESSAGE(stream.good(), "failed to open text fixture: " << path.string());
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

inline void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
inline void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
inline void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
inline void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
inline void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
inline void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

inline bool tokenizer_bind_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

inline bool tokenizer_tokenize_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

inline void initialize_embedding_generator(emel::embeddings::generator::sm & embedding_generator,
                                           emel::error::type & initialize_error,
                                           emel::text::tokenizer::sm & tokenizer) {
  emel::embeddings::generator::event::initialize initialize{
    &tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
  initialize.error_out = &initialize_error;

  REQUIRE(embedding_generator.process_event(initialize));
  CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
}

inline void materialize_tensor_names(loaded_te_fixture & fixture) {
  fixture.model->name_bytes_used = 0u;
  for (uint32_t index = 0u; index < fixture.model->n_tensors; ++index) {
    auto & tensor = fixture.model->tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= fixture.file_bytes.size());
    REQUIRE(fixture.model->name_bytes_used + length <= fixture.model->name_storage.size());

    std::memcpy(fixture.model->name_storage.data() + fixture.model->name_bytes_used,
                fixture.file_bytes.data() + source_offset,
                length);
    tensor.name_offset = fixture.model->name_bytes_used;
    fixture.model->name_bytes_used += static_cast<uint32_t>(length);
  }
}

inline bool load_te_vocab_from_file(const std::filesystem::path & path,
                                    emel::model::data::vocab & vocab_out) {
  std::ifstream stream(path);
  if (!stream.good()) {
    return false;
  }

  std::memset(&vocab_out, 0, sizeof(vocab_out));
  vocab_out.tokenizer_model_id = emel::model::data::tokenizer_model::WPM;
  vocab_out.tokenizer_pre_id = emel::model::data::tokenizer_pre::DEFAULT;
  std::strncpy(vocab_out.tokenizer_model_name.data(),
               "bert",
               vocab_out.tokenizer_model_name.size() - 1u);
  std::strncpy(vocab_out.tokenizer_pre_name.data(),
               "default",
               vocab_out.tokenizer_pre_name.size() - 1u);
  emel::text::tokenizer::detail::apply_tokenizer_model_defaults("bert", vocab_out);
  emel::text::tokenizer::preprocessor::detail::apply_tokenizer_pre_defaults(
      "default", vocab_out);

  std::string line = {};
  uint32_t token_id = 0u;
  uint32_t token_bytes_used = 0u;
  while (std::getline(stream, line)) {
    if (token_id >= emel::model::data::k_max_vocab_tokens) {
      return false;
    }
    if (token_bytes_used + line.size() > emel::model::data::k_max_vocab_bytes) {
      return false;
    }

    if (!line.empty()) {
      std::memcpy(vocab_out.token_storage.data() + token_bytes_used, line.data(), line.size());
    }
    auto & entry = vocab_out.entries[token_id];
    entry.text_offset = token_bytes_used;
    entry.text_length = static_cast<uint32_t>(line.size());
    entry.score = 0.0f;
    entry.type = emel::model::detail::k_token_type_normal;
    token_bytes_used += static_cast<uint32_t>(line.size());
    ++token_id;
  }

  vocab_out.n_tokens = token_id;
  vocab_out.token_bytes_used = token_bytes_used;
  emel::model::detail::mark_special_token_type(
      vocab_out, vocab_out.unk_id, emel::model::detail::k_token_type_unknown);
  emel::model::detail::mark_special_token_type(
      vocab_out, vocab_out.pad_id, emel::model::detail::k_token_type_control);
  emel::model::detail::mark_special_token_type(
      vocab_out, vocab_out.cls_id, emel::model::detail::k_token_type_control);
  emel::model::detail::mark_special_token_type(
      vocab_out, vocab_out.sep_id, emel::model::detail::k_token_type_control);
  emel::model::detail::mark_special_token_type(
      vocab_out, vocab_out.mask_id, emel::model::detail::k_token_type_control);
  return stream.eof();
}

inline loaded_te_fixture load_te_fixture() {
  loaded_te_fixture fixture = {};
  fixture.file_bytes = read_binary_file(te_fixture_path());

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
    std::span<const uint8_t>{fixture.file_bytes},
    requirements,
    on_probe_done,
    on_probe_error,
  };
  REQUIRE(loader.process_event(probe));

  REQUIRE(requirements.tensor_count > 0u);
  REQUIRE(requirements.tensor_count <= fixture.model->tensors.size());

  fixture.kv_arena.resize(
      static_cast<size_t>(emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.model->n_tensors = requirements.tensor_count;

  const emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{fixture.kv_arena},
    std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
    std::span<emel::model::data::tensor_record>{
        fixture.model->tensors.data(), fixture.model->n_tensors},
    on_bind_done,
    on_bind_error,
  };
  REQUIRE(loader.process_event(bind));

  const emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{fixture.file_bytes},
    on_parse_done,
    on_parse_error,
  };
  REQUIRE(loader.process_event(parse));

  const emel::model::detail::kv_binding binding{
    .arena = std::span<const uint8_t>{fixture.kv_arena},
    .entries = std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *fixture.model));
  REQUIRE(load_te_vocab_from_file(te_vocab_path(), fixture.model->vocab_data));

  fixture.model->weights_data = fixture.file_bytes.data();
  fixture.model->weights_size = fixture.file_bytes.size();
  fixture.model->weights_mapped = true;
  materialize_tensor_names(fixture);
  return fixture;
}

inline const loaded_te_fixture & cached_te_fixture() {
  static const loaded_te_fixture fixture = load_te_fixture();
  return fixture;
}

inline float l2_norm(const std::span<const float> values) {
  float sum = 0.0f;
  for (const float value : values) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

inline float max_abs_difference(const std::span<const float> lhs,
                                const std::span<const float> rhs) {
  REQUIRE(lhs.size() == rhs.size());

  float max_diff = 0.0f;
  for (size_t index = 0; index < lhs.size(); ++index) {
    const float diff = std::fabs(lhs[index] - rhs[index]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

inline std::vector<uint8_t> make_rgba_square(const uint8_t r,
                                             const uint8_t g,
                                             const uint8_t b,
                                             const int32_t width,
                                             const int32_t height) {
  std::vector<uint8_t> rgba(static_cast<size_t>(width * height * 4));
  for (int32_t pixel = 0; pixel < width * height; ++pixel) {
    const size_t offset = static_cast<size_t>(pixel) * 4u;
    rgba[offset] = r;
    rgba[offset + 1u] = g;
    rgba[offset + 2u] = b;
    rgba[offset + 3u] = 255u;
  }
  return rgba;
}

inline std::array<float, 4000> make_sine_wave(const float frequency_hz,
                                              const float amplitude = 0.2f,
                                              const int32_t sample_rate = 16000) {
  constexpr float k_pi = 3.14159265358979323846f;

  std::array<float, 4000> samples = {};
  for (int32_t index = 0; index < 4000; ++index) {
    const float phase =
        2.0f * k_pi * frequency_hz * static_cast<float>(index) / static_cast<float>(sample_rate);
    samples[static_cast<size_t>(index)] = amplitude * std::sin(phase);
  }
  return samples;
}

}  // namespace emel::tests::embeddings::te_fixture
