#include <array>
#include <cstdint>
#include <vector>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/detail.hpp"
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
using te_fixture::make_rgba_square;
using te_fixture::max_abs_difference;
using te_fixture::te_assets_present;

struct image_embed_callback_probe {
  bool done_called = false;
  bool error_called = false;
  const emel::embeddings::generator::event::embed_image * request = nullptr;
  int32_t output_dimension = 0;
  emel::error::type err = emel::error::cast(emel::embeddings::generator::error::none);

  void on_done(const emel::embeddings::generator::events::image_embedding_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    output_dimension = ev.output_dimension;
  }

  void on_error(const emel::embeddings::generator::events::image_embedding_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

}  // namespace

TEST_CASE("embeddings vision lane returns normalized TE embeddings when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE vision-lane embedding test because maintained assets are not present");
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

  const std::vector<uint8_t> red_square = make_rgba_square(255u, 0u, 0u, 32, 32);
  const std::vector<uint8_t> blue_square = make_rgba_square(0u, 0u, 255u, 32, 32);

  std::array<float, 1280> red_embedding = {};
  std::array<float, 1280> blue_embedding = {};
  int32_t red_dimension = -1;
  int32_t blue_dimension = -1;
  emel::error::type red_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type blue_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_image red_request{
    red_square,
    32,
    32,
    red_embedding,
    red_dimension,
  };
  red_request.error_out = &red_error;
  REQUIRE(embedding_generator.process_event(red_request));
  CHECK(red_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(red_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            red_embedding.data(),
            static_cast<size_t>(red_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));

  emel::embeddings::generator::event::embed_image blue_request{
    blue_square,
    32,
    32,
    blue_embedding,
    blue_dimension,
  };
  blue_request.error_out = &blue_error;
  REQUIRE(embedding_generator.process_event(blue_request));
  CHECK(blue_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(blue_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            blue_embedding.data(),
            static_cast<size_t>(blue_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{
            red_embedding.data(),
            static_cast<size_t>(red_dimension)},
        std::span<const float>{
            blue_embedding.data(),
            static_cast<size_t>(blue_dimension)}) > 1.0e-5f);
}

TEST_CASE("embeddings vision lane supports truncation and rejects malformed image payloads") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE vision truncation test because maintained assets are not present");
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

  const std::vector<uint8_t> red_square = make_rgba_square(255u, 0u, 0u, 32, 32);
  std::array<float, 256> truncated_output = {};
  int32_t truncated_dimension = -1;
  emel::error::type truncated_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_image truncate_request{
    red_square,
    32,
    32,
    truncated_output,
    truncated_dimension,
  };
  truncate_request.truncate_dimension = 256;
  truncate_request.error_out = &truncated_error;

  REQUIRE(embedding_generator.process_event(truncate_request));
  CHECK(truncated_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(truncated_dimension == 256);
  CHECK(l2_norm(std::span<const float>{truncated_output.data(), 256u}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));

  std::array<float, 1280> invalid_output = {};
  int32_t invalid_dimension = -1;
  emel::error::type invalid_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_image invalid_request{
    std::span<const uint8_t>{red_square.data(), red_square.size() - 1u},
    32,
    32,
    invalid_output,
    invalid_dimension,
  };
  invalid_request.error_out = &invalid_error;

  CHECK_FALSE(embedding_generator.process_event(invalid_request));
  CHECK(invalid_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(invalid_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings vision helper paths cover image request callbacks and validation") {
  CHECK(embedding_detail::is_valid_image_payload(std::span<const uint8_t>{}, 0, 0) == false);

  std::vector<uint8_t> rgba = make_rgba_square(255u, 0u, 0u, 8, 8);
  std::array<float, 1280> output = {};
  int32_t output_dimension = -1;
  emel::error::type embed_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  image_embed_callback_probe probe = {};
  emel::embeddings::generator::event::embed_image request{
    rgba,
    8,
    8,
    output,
    output_dimension,
  };
  request.error_out = &embed_error;
  request.on_done =
      emel::callback<void(const emel::embeddings::generator::events::image_embedding_done &)>::from<
          image_embed_callback_probe,
          &image_embed_callback_probe::on_done>(&probe);
  request.on_error =
      emel::callback<void(const emel::embeddings::generator::events::image_embedding_error &)>::from<
          image_embed_callback_probe,
          &image_embed_callback_probe::on_error>(&probe);

  emel::embeddings::generator::event::embed_image_ctx runtime_ctx = {};
  emel::embeddings::generator::event::embed_image_run runtime_ev{request, runtime_ctx};
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
