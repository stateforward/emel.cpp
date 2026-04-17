#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "te_fixture.hpp"

namespace {

namespace embedding_detail = emel::embeddings::generator::detail;
namespace te_fixture = emel::tests::embeddings::te_fixture;

using te_fixture::cached_te_fixture;
using te_fixture::initialize_embedding_generator;
using te_fixture::inspectable_embedding_generator;
using te_fixture::l2_norm;
using te_fixture::make_rgba_square;
using te_fixture::make_sine_wave;
using te_fixture::max_abs_difference;
using te_fixture::read_text_file;
using te_fixture::te_assets_present;
using te_fixture::te_prompt_path;

inline constexpr int32_t k_audio_sample_rate = 16000;

struct audio_embed_callback_probe {
  bool done_called = false;
  bool error_called = false;
  const emel::embeddings::generator::event::embed_audio * request = nullptr;
  int32_t output_dimension = 0;
  emel::error::type err = emel::error::cast(emel::embeddings::generator::error::none);

  void on_done(const emel::embeddings::generator::events::audio_embedding_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    output_dimension = ev.output_dimension;
  }

  void on_error(const emel::embeddings::generator::events::audio_embedding_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

}  // namespace

TEST_CASE("embeddings audio lane returns normalized TE embeddings when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio-lane embedding test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  REQUIRE(emel::model::architecture_name_view(*fixture.model) == "omniembed");

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

  const auto tone_440 = make_sine_wave(440.0f);
  const auto tone_880 = make_sine_wave(880.0f);

  std::array<float, 1280> embedding_440 = {};
  std::array<float, 1280> embedding_880 = {};
  int32_t dimension_440 = -1;
  int32_t dimension_880 = -1;
  emel::error::type error_440 =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type error_880 =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_audio request_440{
    tone_440,
    k_audio_sample_rate,
    embedding_440,
    dimension_440,
  };
  request_440.error_out = &error_440;
  const bool accepted_440 = embedding_generator.process_event(request_440);
  REQUIRE(accepted_440);
  CHECK(error_440 == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(dimension_440 == 1280);
  CHECK(l2_norm(std::span<const float>{
            embedding_440.data(),
            static_cast<size_t>(dimension_440)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));

  emel::embeddings::generator::event::embed_audio request_880{
    tone_880,
    k_audio_sample_rate,
    embedding_880,
    dimension_880,
  };
  request_880.error_out = &error_880;
  REQUIRE(embedding_generator.process_event(request_880));
  CHECK(error_880 == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(dimension_880 == 1280);
  CHECK(l2_norm(std::span<const float>{
            embedding_880.data(),
            static_cast<size_t>(dimension_880)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{
            embedding_440.data(),
            static_cast<size_t>(dimension_440)},
        std::span<const float>{
            embedding_880.data(),
            static_cast<size_t>(dimension_880)}) > 1.0e-5f);
}

TEST_CASE("embeddings audio detail path keeps prepare and encode outputs finite") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio detail-path test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const auto tone_440 = make_sine_wave(440.0f);

  emel::embeddings::generator::action::context context = {};
  context.model = fixture.model.get();
  REQUIRE(embedding_detail::reserve_scratch(context, *fixture.model));

  const auto is_finite = [](std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](const float value) {
      return std::isfinite(value);
    });
  };

  REQUIRE(embedding_detail::prepare_audio_input(context, tone_440, k_audio_sample_rate));
  const auto prepared = std::span<const float>{
    context.scratch.audio_input.get(),
    static_cast<size_t>(context.audio.num_mel_bins) * static_cast<size_t>(context.audio.time_frames),
  };
  CHECK(is_finite(prepared));

  REQUIRE(embedding_detail::run_audio_embedding(context) ==
          emel::error::cast(emel::embeddings::generator::error::none));
  const auto pooled = std::span<const float>{
    context.scratch.audio_embedding.get(),
    static_cast<size_t>(context.audio.embedding_size),
  };
  CHECK(is_finite(pooled));

  const auto full_embedding = std::span<const float>{
    context.scratch.full_embedding.get(),
    static_cast<size_t>(embedding_detail::shared_embedding_size(context)),
  };
  CHECK(is_finite(full_embedding));
  CHECK(l2_norm(full_embedding) == doctest::Approx(1.0f).epsilon(1.0e-4f));
}

TEST_CASE("embeddings audio runtime sizes feature buffers for pre-stride expand tensors") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio buffer sizing test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();

  emel::embeddings::generator::action::context context = {};
  context.model = fixture.model.get();
  REQUIRE(embedding_detail::reserve_scratch(context, *fixture.model));

  int32_t height = context.audio.num_mel_bins;
  int32_t width = context.audio.time_frames;
  height = embedding_detail::output_dim_same(height, 3, 2);
  width = embedding_detail::output_dim_same(width, 3, 2);
  REQUIRE(context.audio.feature_buffer_elements >=
          height * width * context.audio.stem.output_channels);

  for (int32_t index = 0; index < context.audio.block_count; ++index) {
    const auto & block = context.audio.blocks[static_cast<size_t>(index)];
    CHECK(context.audio.feature_buffer_elements >=
          height * width * block.expanded_channels);

    const int32_t output_height =
        embedding_detail::output_dim_same(height, block.kernel_size, block.stride);
    const int32_t output_width =
        embedding_detail::output_dim_same(width, block.kernel_size, block.stride);
    CHECK(context.audio.feature_buffer_elements >=
          output_height * output_width * block.output_channels);
    height = output_height;
    width = output_width;
  }

  const int32_t head_height = embedding_detail::output_dim_same(height, 1, 1);
  const int32_t head_width = embedding_detail::output_dim_same(width, 1, 1);
  CHECK(context.audio.feature_buffer_elements >= head_height * head_width * 1920);
}

TEST_CASE("embeddings audio lane stays stable after an image request on the same generator") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio-after-image regression test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const auto tone_440 = make_sine_wave(440.0f);
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);

  auto make_generator = [&](emel::text::conditioner::sm & conditioner) {
    return emel::embeddings::generator::sm{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };
  };

  emel::text::tokenizer::sm tokenizer_reference{};
  emel::text::conditioner::sm conditioner_reference{};
  auto reference_generator = make_generator(conditioner_reference);
  emel::error::type initialize_error_reference =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(reference_generator, initialize_error_reference, tokenizer_reference);

  std::array<float, 1280> reference_audio = {};
  int32_t reference_dimension = -1;
  emel::error::type reference_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio reference_request{
    tone_440,
    k_audio_sample_rate,
    reference_audio,
    reference_dimension,
  };
  reference_request.error_out = &reference_error;
  REQUIRE(reference_generator.process_event(reference_request));
  REQUIRE(reference_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(reference_dimension == 1280);

  emel::text::tokenizer::sm tokenizer_shared{};
  emel::text::conditioner::sm conditioner_shared{};
  auto shared_generator = make_generator(conditioner_shared);
  emel::error::type initialize_error_shared =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(shared_generator, initialize_error_shared, tokenizer_shared);

  std::array<float, 1280> image_output = {};
  int32_t image_dimension = -1;
  emel::error::type image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    image_output,
    image_dimension,
  };
  image_request.error_out = &image_error;
  REQUIRE(shared_generator.process_event(image_request));
  REQUIRE(image_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(image_dimension == 1280);

  std::array<float, 1280> audio_after_image = {};
  int32_t audio_after_image_dimension = -1;
  emel::error::type audio_after_image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio audio_after_image_request{
    tone_440,
    k_audio_sample_rate,
    audio_after_image,
    audio_after_image_dimension,
  };
  audio_after_image_request.error_out = &audio_after_image_error;
  REQUIRE(shared_generator.process_event(audio_after_image_request));
  CHECK(audio_after_image_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(audio_after_image_dimension == 1280);
  CHECK(std::all_of(audio_after_image.begin(),
                    audio_after_image.begin() + audio_after_image_dimension,
                    [](const float value) { return std::isfinite(value); }));
  CHECK(l2_norm(std::span<const float>{audio_after_image.data(),
                                      static_cast<size_t>(audio_after_image_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{reference_audio.data(),
                                                 static_cast<size_t>(reference_dimension)},
                           std::span<const float>{audio_after_image.data(),
                                                 static_cast<size_t>(audio_after_image_dimension)}) <=
        1.0e-5f);
}

TEST_CASE("embeddings audio lane surfaces runtime encode failures as backend errors") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio runtime-failure test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const auto tone_440 = make_sine_wave(440.0f);

  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  inspectable_embedding_generator embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  embedding_generator.context_ref().scratch.audio_embedding.reset();

  std::array<float, 1280> output = {};
  int32_t output_dimension = -1;
  emel::error::type embed_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio request{
    tone_440,
    k_audio_sample_rate,
    output,
    output_dimension,
  };
  request.error_out = &embed_error;

  CHECK_FALSE(embedding_generator.process_event(request));
  CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::backend));
  CHECK(output_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings audio lane stays stable after back-to-back text requests") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE text-text audio regression test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::string pure_tone = read_text_file(te_prompt_path("pure-tone-440hz.txt"));
  const std::array red_square_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const std::array pure_tone_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = pure_tone},
  };
  const auto tone_440 = make_sine_wave(440.0f);

  auto make_generator = [&](emel::text::conditioner::sm & conditioner) {
    return emel::embeddings::generator::sm{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };
  };

  emel::text::tokenizer::sm tokenizer_reference{};
  emel::text::conditioner::sm conditioner_reference{};
  auto reference_generator = make_generator(conditioner_reference);
  emel::error::type initialize_error_reference =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(reference_generator, initialize_error_reference, tokenizer_reference);

  std::array<float, 1280> reference_audio = {};
  int32_t reference_dimension = -1;
  emel::error::type reference_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio reference_request{
    tone_440,
    k_audio_sample_rate,
    reference_audio,
    reference_dimension,
  };
  reference_request.error_out = &reference_error;
  REQUIRE(reference_generator.process_event(reference_request));
  REQUIRE(reference_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(reference_dimension == 1280);

  emel::text::tokenizer::sm tokenizer_shared{};
  emel::text::conditioner::sm conditioner_shared{};
  auto shared_generator = make_generator(conditioner_shared);
  emel::error::type initialize_error_shared =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(shared_generator, initialize_error_shared, tokenizer_shared);

  std::array<float, 1280> text_red_output = {};
  std::array<float, 1280> text_tone_output = {};
  std::array<float, 1280> audio_output = {};
  int32_t text_red_dimension = -1;
  int32_t text_tone_dimension = -1;
  int32_t audio_dimension = -1;
  emel::error::type text_red_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type text_tone_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type audio_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_text text_red_request{
    red_square_messages,
    text_red_output,
    text_red_dimension,
  };
  text_red_request.error_out = &text_red_error;
  REQUIRE(shared_generator.process_event(text_red_request));
  REQUIRE(text_red_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_red_dimension == 1280);

  emel::embeddings::generator::event::embed_text text_tone_request{
    pure_tone_messages,
    text_tone_output,
    text_tone_dimension,
  };
  text_tone_request.error_out = &text_tone_error;
  REQUIRE(shared_generator.process_event(text_tone_request));
  REQUIRE(text_tone_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_tone_dimension == 1280);

  emel::embeddings::generator::event::embed_audio audio_request{
    tone_440,
    k_audio_sample_rate,
    audio_output,
    audio_dimension,
  };
  audio_request.error_out = &audio_error;
  REQUIRE(shared_generator.process_event(audio_request));
  CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(audio_dimension == 1280);
  CHECK(std::all_of(audio_output.begin(),
                    audio_output.begin() + audio_dimension,
                    [](const float value) { return std::isfinite(value); }));
  CHECK(l2_norm(std::span<const float>{audio_output.data(),
                                      static_cast<size_t>(audio_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{reference_audio.data(),
                                                 static_cast<size_t>(reference_dimension)},
                           std::span<const float>{audio_output.data(),
                                                 static_cast<size_t>(audio_dimension)}) <=
        1.0e-5f);
}

TEST_CASE("embeddings audio lane stays stable after the canonical text-text-image sequence") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE canonical-sequence audio regression test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::string pure_tone = read_text_file(te_prompt_path("pure-tone-440hz.txt"));
  const std::array red_square_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const std::array pure_tone_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = pure_tone},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto tone_440 = make_sine_wave(440.0f);

  auto make_generator = [&](emel::text::conditioner::sm & conditioner) {
    return emel::embeddings::generator::sm{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };
  };

  emel::text::tokenizer::sm tokenizer_reference{};
  emel::text::conditioner::sm conditioner_reference{};
  auto reference_generator = make_generator(conditioner_reference);
  emel::error::type initialize_error_reference =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(reference_generator, initialize_error_reference, tokenizer_reference);

  std::array<float, 1280> reference_audio = {};
  int32_t reference_dimension = -1;
  emel::error::type reference_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio reference_request{
    tone_440,
    k_audio_sample_rate,
    reference_audio,
    reference_dimension,
  };
  reference_request.error_out = &reference_error;
  REQUIRE(reference_generator.process_event(reference_request));
  REQUIRE(reference_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(reference_dimension == 1280);

  emel::text::tokenizer::sm tokenizer_shared{};
  emel::text::conditioner::sm conditioner_shared{};
  auto shared_generator = make_generator(conditioner_shared);
  emel::error::type initialize_error_shared =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(shared_generator, initialize_error_shared, tokenizer_shared);

  std::array<float, 1280> text_red_output = {};
  std::array<float, 1280> text_tone_output = {};
  std::array<float, 1280> image_output = {};
  std::array<float, 1280> audio_output = {};
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
    text_red_output,
    text_red_dimension,
  };
  text_red_request.error_out = &text_red_error;
  REQUIRE(shared_generator.process_event(text_red_request));
  REQUIRE(text_red_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_red_dimension == 1280);

  emel::embeddings::generator::event::embed_text text_tone_request{
    pure_tone_messages,
    text_tone_output,
    text_tone_dimension,
  };
  text_tone_request.error_out = &text_tone_error;
  REQUIRE(shared_generator.process_event(text_tone_request));
  REQUIRE(text_tone_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(text_tone_dimension == 1280);

  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    image_output,
    image_dimension,
  };
  image_request.error_out = &image_error;
  REQUIRE(shared_generator.process_event(image_request));
  REQUIRE(image_error == emel::error::cast(emel::embeddings::generator::error::none));
  REQUIRE(image_dimension == 1280);
  CHECK(std::all_of(image_output.begin(),
                    image_output.begin() + image_dimension,
                    [](const float value) { return std::isfinite(value); }));
  CHECK(l2_norm(std::span<const float>{image_output.data(),
                                      static_cast<size_t>(image_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));

  emel::embeddings::generator::event::embed_audio audio_request{
    tone_440,
    k_audio_sample_rate,
    audio_output,
    audio_dimension,
  };
  audio_request.error_out = &audio_error;
  REQUIRE(shared_generator.process_event(audio_request));
  CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(audio_dimension == 1280);
  CHECK(std::all_of(audio_output.begin(),
                    audio_output.begin() + audio_dimension,
                    [](const float value) { return std::isfinite(value); }));
  CHECK(l2_norm(std::span<const float>{audio_output.data(),
                                      static_cast<size_t>(audio_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{reference_audio.data(),
                                                 static_cast<size_t>(reference_dimension)},
                           std::span<const float>{audio_output.data(),
                                                 static_cast<size_t>(audio_dimension)}) <=
        1.0e-5f);
}

TEST_CASE("embeddings audio lane supports truncation and rejects malformed audio payloads") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE audio truncation test because maintained assets are not present");
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

  const auto tone_440 = make_sine_wave(440.0f);

  std::array<float, 256> truncated_output = {};
  int32_t truncated_dimension = -1;
  emel::error::type truncated_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio truncate_request{
    tone_440,
    k_audio_sample_rate,
    truncated_output,
    truncated_dimension,
  };
  truncate_request.truncate_dimension = 256;
  truncate_request.error_out = &truncated_error;

  const bool truncate_accepted = embedding_generator.process_event(truncate_request);
  REQUIRE(truncate_accepted);
  CHECK(truncated_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(truncated_dimension == 256);
  CHECK(l2_norm(std::span<const float>{truncated_output.data(), 256u}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));

  std::array<float, 1280> invalid_output = {};
  int32_t invalid_dimension = -1;
  emel::error::type invalid_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio invalid_request{
    tone_440,
    8000,
    invalid_output,
    invalid_dimension,
  };
  invalid_request.error_out = &invalid_error;

  CHECK_FALSE(embedding_generator.process_event(invalid_request));
  CHECK(invalid_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(invalid_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings audio helper paths cover audio request callbacks and validation") {
  CHECK_FALSE(embedding_detail::is_valid_audio_payload(std::span<const float>{}, 0));

  const auto tone_440 = make_sine_wave(440.0f);
  CHECK_FALSE(embedding_detail::is_valid_audio_payload(tone_440, 8000));
  std::array<float, 1280> output = {};
  int32_t output_dimension = -1;
  emel::error::type embed_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  audio_embed_callback_probe probe = {};
  emel::embeddings::generator::event::embed_audio request{
    tone_440,
    k_audio_sample_rate,
    output,
    output_dimension,
  };
  request.error_out = &embed_error;
  request.on_done =
      emel::callback<void(const emel::embeddings::generator::events::audio_embedding_done &)>::from<
          audio_embed_callback_probe,
          &audio_embed_callback_probe::on_done>(&probe);
  request.on_error =
      emel::callback<void(const emel::embeddings::generator::events::audio_embedding_error &)>::from<
          audio_embed_callback_probe,
          &audio_embed_callback_probe::on_error>(&probe);

  emel::embeddings::generator::event::embed_audio_ctx runtime_ctx = {};
  emel::embeddings::generator::event::embed_audio_run runtime_ev{request, runtime_ctx};
  CHECK(embedding_detail::has_embed_callbacks(runtime_ev));
  CHECK(embedding_detail::has_embed_error_callback(runtime_ev));
  runtime_ctx.output_dimension = 321;
  embedding_detail::set_error(runtime_ev, emel::embeddings::generator::error::invalid_request);
  embedding_detail::write_embed_error_out(runtime_ev);
  embedding_detail::emit_embed_done(runtime_ev);
  embedding_detail::emit_embed_error(runtime_ev);

  CHECK(output_dimension == 321);
  CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(probe.done_called);
  CHECK(probe.error_called);
  CHECK(probe.request == &request);
}
