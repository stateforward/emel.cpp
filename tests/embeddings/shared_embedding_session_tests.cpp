#include <array>
#include <string>

#include <boost/sml.hpp>
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
using te_fixture::l2_norm;
using te_fixture::make_rgba_square;
using te_fixture::make_sine_wave;
using te_fixture::read_text_file;
using te_fixture::te_assets_present;
using te_fixture::te_prompt_path;

inline constexpr int32_t k_audio_sample_rate = 16000;

template <size_t capacity>
void check_normalized(const std::array<float, capacity> & output,
                      const int32_t dimension) {
  CHECK(l2_norm(std::span<const float>{output.data(), static_cast<size_t>(dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
}

}  // namespace

TEST_CASE("embeddings shared contract supports the same truncation surface across modalities") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE shared-contract truncation test because maintained assets are not present");
    return;
  }

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

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::array text_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = make_sine_wave(440.0f);
  const std::array<int32_t, 5> supported_dimensions = {1280, 768, 512, 256, 128};

  for (const int32_t dimension : supported_dimensions) {
    std::array<float, 1280> text_output = {};
    std::array<float, 1280> image_output = {};
    std::array<float, 1280> audio_output = {};
    int32_t text_dimension = -1;
    int32_t image_dimension = -1;
    int32_t audio_dimension = -1;
    emel::error::type text_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::error::type image_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::error::type audio_error =
        emel::error::cast(emel::embeddings::generator::error::none);

    emel::embeddings::generator::event::embed_text text_request{
      text_messages,
      std::span<float>{text_output.data(), static_cast<size_t>(dimension)},
      text_dimension,
    };
    text_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    text_request.error_out = &text_error;
    REQUIRE(embedding_generator.process_event(text_request));
    CHECK(text_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(text_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(text_output, text_dimension);

    emel::embeddings::generator::event::embed_image image_request{
      image,
      32,
      32,
      std::span<float>{image_output.data(), static_cast<size_t>(dimension)},
      image_dimension,
    };
    image_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    image_request.error_out = &image_error;
    REQUIRE(embedding_generator.process_event(image_request));
    CHECK(image_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(image_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(image_output, image_dimension);

    emel::embeddings::generator::event::embed_audio audio_request{
      audio,
      k_audio_sample_rate,
      std::span<float>{audio_output.data(), static_cast<size_t>(dimension)},
      audio_dimension,
    };
    audio_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    audio_request.error_out = &audio_error;
    REQUIRE(embedding_generator.process_event(audio_request));
    CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(audio_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(audio_output, audio_dimension);
  }
}

TEST_CASE("embeddings shared contract rejects unsupported truncation uniformly across modalities") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE shared-contract rejection test because maintained assets are not present");
    return;
  }

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

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::array text_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = make_sine_wave(440.0f);

  std::array<float, 1280> text_output = {};
  std::array<float, 1280> image_output = {};
  std::array<float, 1280> audio_output = {};
  int32_t text_dimension = -1;
  int32_t image_dimension = -1;
  int32_t audio_dimension = -1;
  emel::error::type text_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type audio_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_text text_request{
    text_messages,
    text_output,
    text_dimension,
  };
  text_request.truncate_dimension = 640;
  text_request.error_out = &text_error;
  CHECK_FALSE(embedding_generator.process_event(text_request));
  CHECK(text_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(text_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));

  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    image_output,
    image_dimension,
  };
  image_request.truncate_dimension = 640;
  image_request.error_out = &image_error;
  CHECK_FALSE(embedding_generator.process_event(image_request));
  CHECK(image_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(image_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));

  emel::embeddings::generator::event::embed_audio audio_request{
    audio,
    k_audio_sample_rate,
    audio_output,
    audio_dimension,
  };
  audio_request.truncate_dimension = 640;
  audio_request.error_out = &audio_error;
  CHECK_FALSE(embedding_generator.process_event(audio_request));
  CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(audio_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}
