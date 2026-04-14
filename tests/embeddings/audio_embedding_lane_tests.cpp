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
using te_fixture::l2_norm;
using te_fixture::make_sine_wave;
using te_fixture::max_abs_difference;
using te_fixture::te_assets_present;

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
