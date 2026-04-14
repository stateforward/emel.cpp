#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "te_fixture.hpp"

namespace {

namespace te_fixture = emel::tests::embeddings::te_fixture;

using te_fixture::cached_te_fixture;
using te_fixture::initialize_embedding_generator;
using te_fixture::make_rgba_square;
using te_fixture::make_sine_wave;
using te_fixture::read_text_file;
using te_fixture::te_assets_present;
using te_fixture::te_prompt_path;

inline constexpr int32_t k_audio_sample_rate = 16000;
inline constexpr int32_t k_embedding_dimension = 1280;
// The maintained q8 GGUF slice tracks source safetensors most tightly on text,
// with slightly looser but still stable alignment on image/audio.
inline constexpr float k_text_golden_floor = 0.995f;
inline constexpr float k_image_golden_floor = 0.93f;
inline constexpr float k_audio_golden_floor = 0.985f;

struct canonical_embeddings {
  std::array<float, k_embedding_dimension> text_red = {};
  std::array<float, k_embedding_dimension> text_tone = {};
  std::array<float, k_embedding_dimension> image_red = {};
  std::array<float, k_embedding_dimension> audio_tone = {};
};

inline std::filesystem::path te_golden_path(const std::string_view name) {
  return std::filesystem::path{__FILE__}.parent_path() / "fixtures" / "te75m" / std::string{name};
}

inline bool te_goldens_present() {
  return std::filesystem::exists(te_golden_path("red-square.text.1280.txt")) &&
      std::filesystem::exists(te_golden_path("pure-tone-440hz.text.1280.txt")) &&
      std::filesystem::exists(te_golden_path("red-square.image.1280.txt")) &&
      std::filesystem::exists(te_golden_path("pure-tone-440hz.audio.1280.txt"));
}

inline std::vector<float> read_golden_vector(const std::filesystem::path & path) {
  std::ifstream stream(path);
  REQUIRE_MESSAGE(stream.good(), "failed to open golden vector: " << path.string());

  std::vector<float> values = {};
  float value = 0.0f;
  while (stream >> value) {
    values.push_back(value);
  }
  REQUIRE_MESSAGE(stream.eof(), "failed while parsing golden vector: " << path.string());
  REQUIRE(values.size() == static_cast<size_t>(k_embedding_dimension));
  return values;
}

inline float cosine_similarity(const std::span<const float> lhs,
                               const std::span<const float> rhs) {
  REQUIRE(lhs.size() == rhs.size());
  float sum = 0.0f;
  for (size_t index = 0; index < lhs.size(); ++index) {
    sum += lhs[index] * rhs[index];
  }
  return sum;
}

inline canonical_embeddings compute_canonical_embeddings() {
  const auto & fixture = cached_te_fixture();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const std::string red_square_text = read_text_file(te_prompt_path("red-square.txt"));
  const std::string pure_tone_text = read_text_file(te_prompt_path("pure-tone-440hz.txt"));
  const std::array red_square_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square_text},
  };
  const std::array pure_tone_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = pure_tone_text},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = make_sine_wave(440.0f);

  canonical_embeddings outputs = {};
  int32_t text_red_dimension = -1;
  int32_t text_tone_dimension = -1;
  int32_t image_dimension = -1;
  int32_t audio_dimension = -1;
  emel::error::type text_red_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type text_tone_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type audio_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_text text_red_request{
    red_square_messages,
    outputs.text_red,
    text_red_dimension,
  };
  text_red_request.error_out = &text_red_error;
  REQUIRE(embedding_generator.process_event(text_red_request));
  REQUIRE(text_red_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_red_dimension == k_embedding_dimension);

  emel::embeddings::generator::event::embed_text text_tone_request{
    pure_tone_messages,
    outputs.text_tone,
    text_tone_dimension,
  };
  text_tone_request.error_out = &text_tone_error;
  REQUIRE(embedding_generator.process_event(text_tone_request));
  REQUIRE(text_tone_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_tone_dimension == k_embedding_dimension);

  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    outputs.image_red,
    image_dimension,
  };
  image_request.error_out = &image_error;
  REQUIRE(embedding_generator.process_event(image_request));
  REQUIRE(image_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(image_dimension == k_embedding_dimension);

  emel::embeddings::generator::event::embed_audio audio_request{
    audio,
    k_audio_sample_rate,
    outputs.audio_tone,
    audio_dimension,
  };
  audio_request.error_out = &audio_error;
  REQUIRE(embedding_generator.process_event(audio_request));
  REQUIRE(audio_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(audio_dimension == k_embedding_dimension);

  return outputs;
}

inline const canonical_embeddings & cached_canonical_embeddings() {
  static const canonical_embeddings outputs = compute_canonical_embeddings();
  return outputs;
}

}  // namespace

TEST_CASE("TE proof compares EMEL outputs against stored upstream goldens") {
  if (!te_assets_present() || !te_goldens_present()) {
    MESSAGE("skipping TE golden proof test because maintained assets or golden vectors are missing");
    return;
  }

  const auto & outputs = cached_canonical_embeddings();
  const auto golden_text_red = read_golden_vector(te_golden_path("red-square.text.1280.txt"));
  const auto golden_text_tone = read_golden_vector(te_golden_path("pure-tone-440hz.text.1280.txt"));
  const auto golden_image_red = read_golden_vector(te_golden_path("red-square.image.1280.txt"));
  const auto golden_audio_tone = read_golden_vector(te_golden_path("pure-tone-440hz.audio.1280.txt"));

  CHECK(cosine_similarity(outputs.text_red, golden_text_red) >= k_text_golden_floor);
  CHECK(cosine_similarity(outputs.text_tone, golden_text_tone) >= k_text_golden_floor);
  CHECK(cosine_similarity(outputs.image_red, golden_image_red) >= k_image_golden_floor);
  CHECK(cosine_similarity(outputs.audio_tone, golden_audio_tone) >= k_audio_golden_floor);
}

TEST_CASE("TE proof preserves canonical cross-modal smoke relations") {
  if (!te_assets_present() || !te_goldens_present()) {
    MESSAGE("skipping TE cross-modal smoke test because maintained assets or golden vectors are missing");
    return;
  }

  const auto & outputs = cached_canonical_embeddings();
  const auto golden_text_red = read_golden_vector(te_golden_path("red-square.text.1280.txt"));
  const auto golden_text_tone = read_golden_vector(te_golden_path("pure-tone-440hz.text.1280.txt"));
  const auto golden_image_red = read_golden_vector(te_golden_path("red-square.image.1280.txt"));
  const auto golden_audio_tone = read_golden_vector(te_golden_path("pure-tone-440hz.audio.1280.txt"));

  const float text_image = cosine_similarity(outputs.text_red, outputs.image_red);
  const float text_audio_unrelated = cosine_similarity(outputs.text_red, outputs.audio_tone);
  const float tone_audio = cosine_similarity(outputs.text_tone, outputs.audio_tone);

  const float golden_text_image = cosine_similarity(golden_text_red, golden_image_red);
  const float golden_tone_audio = cosine_similarity(golden_text_tone, golden_audio_tone);

  CHECK(text_image > text_audio_unrelated);
  CHECK(text_image > 0.45f);
  CHECK(tone_audio > 0.10f);
  CHECK(std::fabs(text_image - golden_text_image) <= 0.05f);
  CHECK(std::fabs(tone_audio - golden_tone_audio) <= 0.05f);
}
