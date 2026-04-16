#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/detail.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "te_fixture.hpp"

namespace {

namespace embedding_action = emel::embeddings::generator::action;
namespace embedding_detail = emel::embeddings::generator::detail;
namespace te_fixture = emel::tests::embeddings::te_fixture;

using te_fixture::cached_te_fixture;
using te_fixture::l2_norm;
using te_fixture::max_abs_difference;
using te_fixture::read_text_file;
using te_fixture::te_assets_present;
using te_fixture::te_prompt_path;
using te_fixture::tokenizer_bind_dispatch;
using te_fixture::tokenizer_tokenize_dispatch;

struct fake_tokenizer_dispatch_state {
  bool bind_accept = true;
  int32_t bind_error = emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  bool tokenize_accept = true;
  int32_t tokenize_error = emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  std::array<int32_t, 4> token_ids = {};
  int32_t token_count = 1;
  bool saw_bind = false;
  bool saw_tokenize = false;
};

bool fake_tokenizer_bind_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::bind & ev) {
  auto & state = *static_cast<fake_tokenizer_dispatch_state *>(tokenizer_sm);
  state.saw_bind = true;
  if (ev.error_out != nullptr) {
    *ev.error_out = state.bind_error;
  }
  return state.bind_accept;
}

bool fake_tokenizer_tokenize_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::tokenize & ev) {
  auto & state = *static_cast<fake_tokenizer_dispatch_state *>(tokenizer_sm);
  state.saw_tokenize = true;
  if (ev.error_out != nullptr) {
    *ev.error_out = state.tokenize_error;
  }
  if (!state.tokenize_accept) {
    return false;
  }
  if (ev.token_ids_out == nullptr ||
      ev.token_count_out == nullptr ||
      ev.token_capacity < state.token_count) {
    return false;
  }
  std::copy_n(state.token_ids.data(), state.token_count, ev.token_ids_out);
  *ev.token_count_out = state.token_count;
  return true;
}

enum class fake_formatter_mode : uint8_t {
  pass_through = 0u,
  invalid_request = 1u,
  model_invalid = 2u,
  backend = 3u,
};

struct fake_formatter_state {
  fake_formatter_mode mode = fake_formatter_mode::pass_through;
};

bool fake_formatter(void * formatter_ctx,
                    const emel::text::formatter::format_request & request,
                    int32_t * error_out) noexcept {
  auto & state = *static_cast<fake_formatter_state *>(formatter_ctx);
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }

  switch (state.mode) {
    case fake_formatter_mode::pass_through:
      return emel::text::formatter::format_raw(nullptr, request, error_out);
    case fake_formatter_mode::invalid_request:
      if (error_out != nullptr) {
        *error_out =
            embedding_detail::conditioner_error_code(emel::text::conditioner::error::invalid_argument);
      }
      return false;
    case fake_formatter_mode::model_invalid:
      if (error_out != nullptr) {
        *error_out =
            embedding_detail::conditioner_error_code(emel::text::conditioner::error::model_invalid);
      }
      return false;
    case fake_formatter_mode::backend:
      if (error_out != nullptr) {
        *error_out =
            embedding_detail::conditioner_error_code(emel::text::conditioner::error::none);
      }
      return false;
  }

  return false;
}

struct initialize_callback_probe {
  bool done_called = false;
  bool error_called = false;
  const emel::embeddings::generator::event::initialize * request = nullptr;
  emel::error::type err = emel::error::cast(emel::embeddings::generator::error::none);

  void on_done(const emel::embeddings::generator::events::initialize_done & ev) noexcept {
    done_called = true;
    request = ev.request;
  }

  void on_error(const emel::embeddings::generator::events::initialize_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

struct embed_callback_probe {
  bool done_called = false;
  bool error_called = false;
  const emel::embeddings::generator::event::embed_text * request = nullptr;
  int32_t output_dimension = 0;
  emel::error::type err = emel::error::cast(emel::embeddings::generator::error::none);

  void on_done(const emel::embeddings::generator::events::text_embedding_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    output_dimension = ev.output_dimension;
  }

  void on_error(const emel::embeddings::generator::events::text_embedding_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

}  // namespace

TEST_CASE("embeddings text lane returns normalized TE embeddings when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE text-lane embedding test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  REQUIRE(emel::model::architecture_name_view(*fixture.model) == "omniembed");
  CHECK(fixture.model->params.n_embd_out == 1280);

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
  emel::embeddings::generator::event::initialize initialize{
    &tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
  initialize.add_special = true;
  initialize.parse_special = false;
  initialize.error_out = &initialize_error;

  REQUIRE(embedding_generator.process_event(initialize));
  CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_idle>));

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::string pure_tone = read_text_file(te_prompt_path("pure-tone-440hz.txt"));
  const std::array red_square_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const std::array pure_tone_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = pure_tone},
  };

  std::array<float, 1280> red_square_embedding = {};
  std::array<float, 1280> pure_tone_embedding = {};
  int32_t red_square_dimension = -1;
  int32_t pure_tone_dimension = -1;
  emel::error::type red_square_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type pure_tone_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_text red_square_request{
    red_square_messages,
    red_square_embedding,
    red_square_dimension,
  };
  red_square_request.error_out = &red_square_error;
  REQUIRE(embedding_generator.process_event(red_square_request));
  CHECK(red_square_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(red_square_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            red_square_embedding.data(),
            static_cast<size_t>(red_square_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));

  emel::embeddings::generator::event::embed_text pure_tone_request{
    pure_tone_messages,
    pure_tone_embedding,
    pure_tone_dimension,
  };
  pure_tone_request.error_out = &pure_tone_error;
  REQUIRE(embedding_generator.process_event(pure_tone_request));
  CHECK(pure_tone_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(pure_tone_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            pure_tone_embedding.data(),
            static_cast<size_t>(pure_tone_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));

  CHECK(max_abs_difference(std::span<const float>{
            red_square_embedding.data(),
            static_cast<size_t>(red_square_dimension)},
        std::span<const float>{
            pure_tone_embedding.data(),
            static_cast<size_t>(pure_tone_dimension)}) > 1.0e-5f);
}

TEST_CASE("embeddings generator initializes with TE q5 fixture when present") {
  const auto q5_fixture_path =
      te_fixture::repo_root() / "tests" / "models" / "TE-75M-q5_0.gguf";
  if (!std::filesystem::exists(q5_fixture_path) || !std::filesystem::exists(te_fixture::te_vocab_path())) {
    MESSAGE("skipping TE q5 initialize test because maintained q5 assets are not present");
    return;
  }

  const auto fixture = te_fixture::load_te_fixture(q5_fixture_path);
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
  emel::embeddings::generator::event::initialize initialize{
    &tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
  initialize.add_special = true;
  initialize.parse_special = false;
  initialize.error_out = &initialize_error;

  REQUIRE(embedding_generator.process_event(initialize));
  CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_idle>));
}

TEST_CASE("maintained TE fixture selector approves q8 and q5 only") {
  const auto q8_path = te_fixture::repo_root() / "tests" / "models" / "TE-75M-q8_0.gguf";
  const auto q5_path = te_fixture::repo_root() / "tests" / "models" / "TE-75M-q5_0.gguf";
  const auto q4_path = te_fixture::repo_root() / "tests" / "models" / "TE-75M-q4_0.gguf";

  CHECK(te_fixture::is_approved_te_fixture_path(q8_path));
  CHECK(te_fixture::is_approved_te_fixture_path(q5_path));
  CHECK_FALSE(te_fixture::is_approved_te_fixture_path(q4_path));
}

TEST_CASE("embeddings text lane supports TE matryoshka truncation when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE truncation test because maintained assets are not present");
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

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::array messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const std::array<int32_t, 4> supported_dimensions = {768, 512, 256, 128};
  for (const int32_t dimension : supported_dimensions) {
    std::array<float, 768> output = {};
    int32_t output_dimension = -1;
    emel::error::type embed_error =
        emel::error::cast(emel::embeddings::generator::error::none);

    emel::embeddings::generator::event::embed_text request{
      messages,
      std::span<float>{output.data(), static_cast<size_t>(dimension)},
      output_dimension,
    };
    request.truncate_dimension = dimension;
    request.error_out = &embed_error;

    REQUIRE(embedding_generator.process_event(request));
    CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(output_dimension == dimension);
    CHECK(l2_norm(std::span<const float>{output.data(), static_cast<size_t>(dimension)}) ==
          doctest::Approx(1.0f).epsilon(1.0e-4f));
  }

  std::array<float, 768> invalid_output = {};
  int32_t invalid_output_dimension = -1;
  emel::error::type invalid_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_text invalid_request{
    messages,
    invalid_output,
    invalid_output_dimension,
  };
  invalid_request.truncate_dimension = 640;
  invalid_request.error_out = &invalid_error;

  CHECK_FALSE(embedding_generator.process_event(invalid_request));
  CHECK(invalid_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(invalid_output_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings generator helper paths cover tensor binding callbacks and publication") {
  if (!te_assets_present()) {
    MESSAGE("skipping embedding helper coverage test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();

  std::array<float, 4> vector_data = {{1.0f, 2.0f, 3.0f, 4.0f}};
  emel::model::data::tensor_record vector_tensor = {};
  vector_tensor.data = vector_data.data();
  vector_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
  vector_tensor.n_dims = 1;
  vector_tensor.dims[0] = 4;

  embedding_action::vector_view vector_view = {};
  CHECK(embedding_detail::bind_vector_f32(vector_tensor, 4, vector_view));
  CHECK(vector_view.data == vector_data.data());
  vector_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  CHECK_FALSE(embedding_detail::bind_vector_f32(vector_tensor, 4, vector_view));

  std::array<float, 6> matrix_data = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
  emel::model::data::tensor_record matrix_tensor = {};
  matrix_tensor.data = matrix_data.data();
  matrix_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
  matrix_tensor.n_dims = 2;
  matrix_tensor.dims[0] = 3;
  matrix_tensor.dims[1] = 2;

  embedding_action::matrix_view matrix_view = {};
  CHECK(embedding_detail::bind_matrix(matrix_tensor, 2, 3, matrix_view));
  CHECK(matrix_view.rows == 2);
  CHECK(matrix_view.cols == 3);
  CHECK(embedding_detail::dense_row_bytes(emel::kernel::detail::dtype_f32, 3) == 3u * sizeof(float));
  CHECK(embedding_detail::dense_row_bytes(emel::kernel::detail::dtype_f16, 3) == 3u * sizeof(uint16_t));
  CHECK(embedding_detail::dense_row_bytes(emel::kernel::detail::dtype_q5_0, 32) ==
        sizeof(emel::kernel::detail::quant::block_q5_0));
  CHECK_FALSE(embedding_detail::bind_matrix(matrix_tensor, 3, 3, matrix_view));

  std::array<char, 256> name_buffer = {};
  std::string_view layer_name = {};
  CHECK(embedding_detail::make_layer_name(3, "attention.self.query.weight", name_buffer, layer_name));
  CHECK(layer_name.find("layer.3.") != std::string_view::npos);
  CHECK_FALSE(embedding_detail::make_layer_name(
      1, std::string(400u, 'x'), name_buffer, layer_name));

  embedding_action::context context = {};
  context.model = fixture.model.get();
  REQUIRE(embedding_detail::reserve_scratch(context, *fixture.model));
  CHECK(embedding_detail::is_valid_preprocessor(
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm));
  CHECK_FALSE(embedding_detail::is_valid_preprocessor(
      static_cast<emel::text::tokenizer::preprocessor::preprocessor_kind>(999)));
  CHECK(embedding_detail::is_valid_encoder(emel::text::encoders::encoder_kind::wpm));
  CHECK_FALSE(embedding_detail::is_valid_encoder(
      static_cast<emel::text::encoders::encoder_kind>(999)));
  CHECK(embedding_detail::is_supported_truncate_dimension(context, 768));
  CHECK(embedding_detail::is_supported_truncate_dimension(context, context.text.shared_embedding_size));
  CHECK_FALSE(embedding_detail::is_supported_truncate_dimension(context, 640));
  CHECK_FALSE(embedding_detail::is_supported_truncate_dimension(context, -1));

  for (int32_t index = 0; index < context.text.shared_embedding_size; ++index) {
    context.scratch.full_embedding[static_cast<size_t>(index)] = static_cast<float>(index + 1);
  }

  std::array<float, 1280> full_output = {};
  std::array<float, 768> truncated_output = {};
  CHECK(embedding_detail::publish_embedding(context, context.text.shared_embedding_size, full_output));
  CHECK(full_output[0] == doctest::Approx(1.0f));
  CHECK(embedding_detail::publish_embedding(context, 768, truncated_output));
  CHECK(l2_norm(truncated_output) == doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK_FALSE(embedding_detail::publish_embedding(context, 0, truncated_output));

  fake_tokenizer_dispatch_state fake_tokenizer = {};
  fake_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_callback_probe initialize_probe = {};
  emel::embeddings::generator::event::initialize initialize{
    &fake_tokenizer,
    fake_tokenizer_bind_dispatch,
    fake_tokenizer_tokenize_dispatch,
  };
  initialize.error_out = &initialize_error;
  initialize.on_done = emel::callback<void(const emel::embeddings::generator::events::initialize_done &)>::from<
      initialize_callback_probe,
      &initialize_callback_probe::on_done>(&initialize_probe);
  initialize.on_error =
      emel::callback<void(const emel::embeddings::generator::events::initialize_error &)>::from<
          initialize_callback_probe,
          &initialize_callback_probe::on_error>(&initialize_probe);

  emel::embeddings::generator::event::initialize_ctx initialize_ctx = {};
  emel::embeddings::generator::event::initialize_run initialize_run{initialize, initialize_ctx};
  CHECK(embedding_detail::has_initialize_callback(initialize_run));
  CHECK(embedding_detail::has_initialize_error_callback(initialize_run));
  embedding_detail::set_error(
      initialize_run, emel::embeddings::generator::error::backend);
  embedding_detail::write_initialize_error_out(initialize_run);
  CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::backend));
  embedding_detail::emit_initialize_done(initialize_run);
  embedding_detail::emit_initialize_error(initialize_run);
  CHECK(initialize_probe.done_called);
  CHECK(initialize_probe.error_called);
  CHECK(initialize_probe.request == &initialize);

  const std::array messages = {
    emel::text::formatter::chat_message{.role = "user", .content = "helper coverage"},
  };
  std::array<float, 1280> output = {};
  int32_t output_dimension = -1;
  emel::error::type embed_error = emel::error::cast(emel::embeddings::generator::error::none);
  embed_callback_probe embed_probe = {};
  emel::embeddings::generator::event::embed_text embed{messages, output, output_dimension};
  embed.error_out = &embed_error;
  embed.on_done = emel::callback<void(const emel::embeddings::generator::events::text_embedding_done &)>::from<
      embed_callback_probe,
      &embed_callback_probe::on_done>(&embed_probe);
  embed.on_error = emel::callback<void(const emel::embeddings::generator::events::text_embedding_error &)>::from<
      embed_callback_probe,
      &embed_callback_probe::on_error>(&embed_probe);

  emel::embeddings::generator::event::embed_text_ctx embed_ctx = {};
  emel::embeddings::generator::event::embed_text_run embed_run{embed, embed_ctx};
  CHECK(embedding_detail::has_embed_callbacks(embed_run));
  CHECK(embedding_detail::has_embed_error_callback(embed_run));
  embed_ctx.output_dimension = 123;
  embedding_detail::set_error(embed_run, emel::embeddings::generator::error::invalid_request);
  embedding_detail::write_embed_error_out(embed_run);
  CHECK(output_dimension == 123);
  CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  embedding_detail::emit_embed_done(embed_run);
  embedding_detail::emit_embed_error(embed_run);
  CHECK(embed_probe.done_called);
  CHECK(embed_probe.error_called);
  CHECK(embed_probe.request == &embed);
}

TEST_CASE("embeddings generator numeric helpers handle valid and invalid inputs") {
  using emel::kernel::detail::dtype_f16;
  using emel::kernel::detail::dtype_f32;
  using emel::kernel::detail::dtype_q5_0;
  using emel::kernel::detail::dtype_q8_0;
  using emel::kernel::detail::quant::QK5_0;
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q5_0;
  using emel::kernel::detail::quant::block_q8_0;
  using emel::kernel::detail::quant::fp32_to_fp16;
  using emel::kernel::detail::quant::quantize_row_q5_0_ref;
  using emel::kernel::detail::quant::quantize_row_q8_0_strided;

  std::array<float, 3> input = {{1.0f, 1.0f, 1.0f}};
  std::array<float, 2> output = {};
  std::array<float, 6> f32_weights = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
  embedding_action::matrix_view f32_matrix = {
    .data = f32_weights.data(),
    .dtype = dtype_f32,
    .rows = 2,
    .cols = 3,
    .row_bytes = 3u * sizeof(float),
  };

  std::array<block_q8_0, 1> q8_scratch = {};
  CHECK(embedding_detail::matmul_f32(f32_matrix, input, output));
  CHECK(output[0] == doctest::Approx(6.0f));
  CHECK(output[1] == doctest::Approx(15.0f));
  CHECK(embedding_detail::matmul(f32_matrix, input, q8_scratch, output));
  CHECK_FALSE(embedding_detail::matmul_f32(
      f32_matrix, std::span<const float>{input.data(), 2u}, output));

  std::array<uint16_t, 6> f16_weights = {{
      fp32_to_fp16(1.0f),
      fp32_to_fp16(2.0f),
      fp32_to_fp16(3.0f),
      fp32_to_fp16(4.0f),
      fp32_to_fp16(5.0f),
      fp32_to_fp16(6.0f),
  }};
  embedding_action::matrix_view f16_matrix = {
    .data = f16_weights.data(),
    .dtype = dtype_f16,
    .rows = 2,
    .cols = 3,
    .row_bytes = 3u * sizeof(uint16_t),
  };
  CHECK(embedding_detail::matmul_f16(f16_matrix, input, output));
  CHECK(output[0] == doctest::Approx(6.0f).epsilon(1.0e-3f));
  CHECK(embedding_detail::matmul(f16_matrix, input, q8_scratch, output));
  CHECK_FALSE(embedding_detail::matmul_f16(
      f16_matrix, input, std::span<float>{output.data(), 1u}));

  std::array<float, QK8_0> q8_row = {};
  q8_row.fill(1.0f);
  std::array<block_q8_0, 1> q8_row_storage = {};
  quantize_row_q8_0_strided(q8_row.data(), 1u, q8_row_storage.data(), QK8_0);
  embedding_action::matrix_view q8_matrix = {
    .data = q8_row_storage.data(),
    .dtype = dtype_q8_0,
    .rows = 1,
    .cols = QK8_0,
    .row_bytes = sizeof(block_q8_0),
  };
  std::array<float, 1> q8_output = {};
  CHECK(embedding_detail::matmul_q8_0(q8_matrix, q8_row, q8_scratch, q8_output));
  CHECK(q8_output[0] == doctest::Approx(static_cast<float>(QK8_0)).epsilon(1.0e-1f));
  CHECK(embedding_detail::matmul(q8_matrix, q8_row, q8_scratch, q8_output));
  std::array<float, QK8_0> dequantized = {};
  CHECK(embedding_detail::copy_embedding_row(q8_matrix, 0, dequantized));
  CHECK(dequantized[0] == doctest::Approx(1.0f).epsilon(2.0e-1f));
  CHECK_FALSE(embedding_detail::copy_embedding_row(
      q8_matrix, 1, dequantized));

  std::array<float, QK5_0> q5_row = {};
  q5_row.fill(1.0f);
  std::array<block_q5_0, 1> q5_row_storage = {};
  quantize_row_q5_0_ref(q5_row.data(), q5_row_storage.data(), QK5_0);
  embedding_action::matrix_view q5_matrix = {
    .data = q5_row_storage.data(),
    .dtype = dtype_q5_0,
    .rows = 1,
    .cols = QK5_0,
    .row_bytes = sizeof(block_q5_0),
  };
  std::array<float, 1> q5_output = {};
  CHECK(embedding_detail::matmul_q5_0(q5_matrix, q5_row, q8_scratch, q5_output));
  CHECK(q5_output[0] == doctest::Approx(static_cast<float>(QK5_0)).epsilon(3.0e-1f));
  CHECK(embedding_detail::matmul(q5_matrix, q5_row, q8_scratch, q5_output));
  std::array<float, QK5_0> q5_dequantized = {};
  CHECK(embedding_detail::copy_embedding_row(q5_matrix, 0, q5_dequantized));
  CHECK(q5_dequantized[0] == doctest::Approx(1.0f).epsilon(2.0e-1f));
  CHECK_FALSE(embedding_detail::copy_embedding_row(q5_matrix, 1, q5_dequantized));

  std::array<float, 2> bias_data = {{0.5f, -0.5f}};
  embedding_action::vector_view bias_view = {
    .data = bias_data.data(),
    .size = 2,
  };
  std::array<float, 2> add_values = {{1.0f, 2.0f}};
  CHECK(embedding_detail::add_bias(add_values, bias_view));
  CHECK(add_values[0] == doctest::Approx(1.5f));
  CHECK(add_values[1] == doctest::Approx(1.5f));
  CHECK_FALSE(embedding_detail::add_bias(
      std::span<float>{add_values.data(), 1u}, bias_view));

  std::array<float, 2> add_dst = {{1.0f, 2.0f}};
  std::array<float, 2> add_src = {{3.0f, 4.0f}};
  CHECK(embedding_detail::add_in_place(add_dst, add_src));
  CHECK(add_dst[0] == doctest::Approx(4.0f));
  CHECK(add_dst[1] == doctest::Approx(6.0f));
  CHECK_FALSE(embedding_detail::add_in_place(
      add_dst, std::span<const float>{add_src.data(), 1u}));

  std::array<float, 2> norm_input = {{1.0f, -1.0f}};
  std::array<float, 2> norm_output = {};
  std::array<float, 2> norm_weights = {{1.0f, 1.0f}};
  std::array<float, 2> norm_bias = {{0.0f, 0.0f}};
  embedding_action::vector_view norm_weight_view = {
    .data = norm_weights.data(),
    .size = 2,
  };
  embedding_action::vector_view norm_bias_view = {
    .data = norm_bias.data(),
    .size = 2,
  };
  CHECK(embedding_detail::layer_norm(norm_input, norm_weight_view, norm_bias_view, 1.0e-5f, norm_output));
  CHECK_FALSE(embedding_detail::layer_norm(
      std::span<const float>{}, norm_weight_view, norm_bias_view, 1.0e-5f, norm_output));

  std::array<float, 2> softmax_values = {{0.0f, 0.0f}};
  CHECK(embedding_detail::soft_max(softmax_values));
  CHECK(softmax_values[0] == doctest::Approx(0.5f));
  CHECK(softmax_values[1] == doctest::Approx(0.5f));
  CHECK_FALSE(embedding_detail::soft_max(std::span<float>{}));

  std::array<float, 2> normalize_values = {{3.0f, 4.0f}};
  CHECK(embedding_detail::l2_normalize(normalize_values));
  CHECK(l2_norm(normalize_values) == doctest::Approx(1.0f).epsilon(1.0e-4f));
  std::array<float, 2> zero_values = {{0.0f, 0.0f}};
  CHECK_FALSE(embedding_detail::l2_normalize(zero_values));
  CHECK_FALSE(embedding_detail::l2_normalize(std::span<float>{}));
  CHECK(embedding_detail::gelu(0.0f) == doctest::Approx(0.0f));
  embedding_detail::apply_gelu(normalize_values);
}

TEST_CASE("embeddings generator state machine covers callback and prepare error branches when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping embedding state-machine coverage test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  const std::array messages = {
    emel::text::formatter::chat_message{.role = "user", .content = "callback coverage"},
  };

  SUBCASE("initialize and embed done callbacks fire") {
    fake_tokenizer_dispatch_state fake_tokenizer = {};
    fake_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
    fake_tokenizer.token_count = 1;
    emel::text::conditioner::sm conditioner{};
    emel::embeddings::generator::sm embedding_generator{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };

    emel::error::type initialize_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    initialize_callback_probe initialize_probe = {};
    emel::embeddings::generator::event::initialize initialize{
      &fake_tokenizer,
      fake_tokenizer_bind_dispatch,
      fake_tokenizer_tokenize_dispatch,
    };
    initialize.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
    initialize.error_out = &initialize_error;
    initialize.on_done =
        emel::callback<void(const emel::embeddings::generator::events::initialize_done &)>::from<
            initialize_callback_probe,
            &initialize_callback_probe::on_done>(&initialize_probe);
    initialize.on_error =
        emel::callback<void(const emel::embeddings::generator::events::initialize_error &)>::from<
            initialize_callback_probe,
            &initialize_callback_probe::on_error>(&initialize_probe);

    REQUIRE(embedding_generator.process_event(initialize));
    CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(initialize_probe.done_called);
    CHECK_FALSE(initialize_probe.error_called);
    CHECK(initialize_probe.request == &initialize);
    CHECK(fake_tokenizer.saw_bind);

    std::array<float, 1280> output = {};
    int32_t output_dimension = -1;
    emel::error::type embed_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    embed_callback_probe embed_probe = {};
    emel::embeddings::generator::event::embed_text request{
      messages,
      output,
      output_dimension,
    };
    request.error_out = &embed_error;
    request.on_done =
        emel::callback<void(const emel::embeddings::generator::events::text_embedding_done &)>::from<
            embed_callback_probe,
            &embed_callback_probe::on_done>(&embed_probe);
    request.on_error =
        emel::callback<void(const emel::embeddings::generator::events::text_embedding_error &)>::from<
            embed_callback_probe,
            &embed_callback_probe::on_error>(&embed_probe);

    REQUIRE(embedding_generator.process_event(request));
    CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(embed_probe.done_called);
    CHECK_FALSE(embed_probe.error_called);
    CHECK(embed_probe.request == &request);
    CHECK(embed_probe.output_dimension == 1280);
    CHECK(fake_tokenizer.saw_tokenize);
  }

  SUBCASE("initialize invalid request uses error callback path") {
    emel::text::conditioner::sm conditioner{};
    emel::embeddings::generator::sm embedding_generator{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };

    emel::error::type initialize_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    initialize_callback_probe initialize_probe = {};
    emel::embeddings::generator::event::initialize initialize{
      nullptr,
      fake_tokenizer_bind_dispatch,
      fake_tokenizer_tokenize_dispatch,
    };
    initialize.error_out = &initialize_error;
    initialize.on_error =
        emel::callback<void(const emel::embeddings::generator::events::initialize_error &)>::from<
            initialize_callback_probe,
            &initialize_callback_probe::on_error>(&initialize_probe);

    CHECK_FALSE(embedding_generator.process_event(initialize));
    CHECK(initialize_error ==
          emel::error::cast(emel::embeddings::generator::error::invalid_request));
    CHECK_FALSE(initialize_probe.done_called);
    CHECK(initialize_probe.error_called);
    CHECK(initialize_probe.request == &initialize);
  }

  SUBCASE("unexpected embed before initialize rejects without trapping later requests") {
    fake_tokenizer_dispatch_state fake_tokenizer = {};
    fake_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
    fake_tokenizer.token_count = 1;
    emel::text::conditioner::sm conditioner{};
    emel::embeddings::generator::sm embedding_generator{
      *fixture.model,
      conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };

    std::array<float, 1280> preinit_output = {};
    int32_t preinit_output_dimension = -1;
    emel::error::type preinit_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::embed_text preinit_request{
      messages,
      preinit_output,
      preinit_output_dimension,
    };
    preinit_request.error_out = &preinit_error;

    CHECK_FALSE(embedding_generator.process_event(preinit_request));
    CHECK(preinit_error ==
          emel::error::cast(emel::embeddings::generator::error::invalid_request));
    CHECK(preinit_output_dimension == 0);
    CHECK(embedding_generator.is(
        boost::sml::state<emel::embeddings::generator::state_uninitialized>));

    emel::error::type initialize_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::initialize initialize{
      &fake_tokenizer,
      fake_tokenizer_bind_dispatch,
      fake_tokenizer_tokenize_dispatch,
    };
    initialize.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
    initialize.error_out = &initialize_error;

    REQUIRE(embedding_generator.process_event(initialize));
    CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_idle>));

    std::array<float, 1280> output = {};
    int32_t output_dimension = -1;
    emel::error::type embed_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::embed_text request{
      messages,
      output,
      output_dimension,
    };
    request.error_out = &embed_error;

    REQUIRE(embedding_generator.process_event(request));
    CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(output_dimension == 1280);
    CHECK(fake_tokenizer.saw_bind);
    CHECK(fake_tokenizer.saw_tokenize);
  }

  SUBCASE("initialize maps tokenizer bind errors to model_invalid and backend") {
    fake_tokenizer_dispatch_state model_invalid_tokenizer = {};
    model_invalid_tokenizer.bind_accept = false;
    model_invalid_tokenizer.bind_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::model_invalid);
    emel::text::conditioner::sm model_invalid_conditioner{};
    emel::embeddings::generator::sm model_invalid_generator{
      *fixture.model,
      model_invalid_conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };

    emel::error::type model_invalid_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::initialize model_invalid_initialize{
      &model_invalid_tokenizer,
      fake_tokenizer_bind_dispatch,
      fake_tokenizer_tokenize_dispatch,
    };
    model_invalid_initialize.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    model_invalid_initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
    model_invalid_initialize.error_out = &model_invalid_error;
    CHECK_FALSE(model_invalid_generator.process_event(model_invalid_initialize));
    CHECK(model_invalid_error ==
          emel::error::cast(emel::embeddings::generator::error::model_invalid));

    fake_tokenizer_dispatch_state backend_tokenizer = {};
    backend_tokenizer.bind_accept = false;
    backend_tokenizer.bind_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::backend_error);
    emel::text::conditioner::sm backend_conditioner{};
    emel::embeddings::generator::sm backend_generator{
      *fixture.model,
      backend_conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };

    emel::error::type backend_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::initialize backend_initialize{
      &backend_tokenizer,
      fake_tokenizer_bind_dispatch,
      fake_tokenizer_tokenize_dispatch,
    };
    backend_initialize.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    backend_initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
    backend_initialize.error_out = &backend_error;
    CHECK_FALSE(backend_generator.process_event(backend_initialize));
    CHECK(backend_error == emel::error::cast(emel::embeddings::generator::error::backend));
  }

  SUBCASE("embed request covers invalid request and conditioning error paths") {
    auto make_initialized_generator = [&](fake_tokenizer_dispatch_state & tokenizer_state,
                                          fake_formatter_state * formatter_state,
                                          emel::text::conditioner::sm & conditioner,
                                          emel::embeddings::generator::sm & generator) {
      emel::error::type initialize_error =
          emel::error::cast(emel::embeddings::generator::error::none);
      emel::embeddings::generator::event::initialize initialize{
        &tokenizer_state,
        fake_tokenizer_bind_dispatch,
        fake_tokenizer_tokenize_dispatch,
      };
      initialize.preprocessor_variant =
          emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
      initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
      initialize.error_out = &initialize_error;
      REQUIRE(generator.process_event(initialize));
      CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
      (void) formatter_state;
      (void) conditioner;
    };

    fake_tokenizer_dispatch_state invalid_tokenizer = {};
    invalid_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
    invalid_tokenizer.token_count = 1;
    emel::text::conditioner::sm invalid_conditioner{};
    emel::embeddings::generator::sm invalid_generator{
      *fixture.model,
      invalid_conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };
    make_initialized_generator(invalid_tokenizer, nullptr, invalid_conditioner, invalid_generator);

    std::array<float, 1280> invalid_output = {};
    int32_t invalid_output_dimension = -1;
    emel::error::type invalid_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    embed_callback_probe invalid_probe = {};
    emel::embeddings::generator::event::embed_text invalid_request{
      messages,
      invalid_output,
      invalid_output_dimension,
    };
    invalid_request.truncate_dimension = 640;
    invalid_request.error_out = &invalid_error;
    invalid_request.on_error =
        emel::callback<void(const emel::embeddings::generator::events::text_embedding_error &)>::from<
            embed_callback_probe,
            &embed_callback_probe::on_error>(&invalid_probe);
    CHECK_FALSE(invalid_generator.process_event(invalid_request));
    CHECK(invalid_error ==
          emel::error::cast(emel::embeddings::generator::error::invalid_request));
    CHECK(invalid_probe.error_called);
    CHECK(invalid_probe.request == &invalid_request);

    fake_tokenizer_dispatch_state formatter_invalid_tokenizer = {};
    formatter_invalid_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
    fake_formatter_state formatter_invalid_state = {.mode = fake_formatter_mode::invalid_request};
    emel::text::conditioner::sm formatter_invalid_conditioner{};
    emel::embeddings::generator::sm formatter_invalid_generator{
      *fixture.model,
      formatter_invalid_conditioner,
      &formatter_invalid_state,
      fake_formatter,
    };
    make_initialized_generator(
        formatter_invalid_tokenizer,
        &formatter_invalid_state,
        formatter_invalid_conditioner,
        formatter_invalid_generator);

    std::array<float, 1280> formatter_invalid_output = {};
    int32_t formatter_invalid_dimension = -1;
    emel::error::type formatter_invalid_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::embed_text formatter_invalid_request{
      messages,
      formatter_invalid_output,
      formatter_invalid_dimension,
    };
    formatter_invalid_request.error_out = &formatter_invalid_error;
    CHECK_FALSE(formatter_invalid_generator.process_event(formatter_invalid_request));
    CHECK(formatter_invalid_error ==
          emel::error::cast(emel::embeddings::generator::error::invalid_request));

    fake_tokenizer_dispatch_state formatter_backend_tokenizer = {};
    formatter_backend_tokenizer.token_ids[0] = fixture.model->vocab_data.cls_id;
    fake_formatter_state formatter_backend_state = {.mode = fake_formatter_mode::backend};
    emel::text::conditioner::sm formatter_backend_conditioner{};
    emel::embeddings::generator::sm formatter_backend_generator{
      *fixture.model,
      formatter_backend_conditioner,
      &formatter_backend_state,
      fake_formatter,
    };
    make_initialized_generator(
        formatter_backend_tokenizer,
        &formatter_backend_state,
        formatter_backend_conditioner,
        formatter_backend_generator);

    std::array<float, 1280> formatter_backend_output = {};
    int32_t formatter_backend_dimension = -1;
    emel::error::type formatter_backend_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::embed_text formatter_backend_request{
      messages,
      formatter_backend_output,
      formatter_backend_dimension,
    };
    formatter_backend_request.error_out = &formatter_backend_error;
    CHECK_FALSE(formatter_backend_generator.process_event(formatter_backend_request));
    CHECK(formatter_backend_error ==
          emel::error::cast(emel::embeddings::generator::error::backend));

    fake_tokenizer_dispatch_state model_invalid_tokenizer = {};
    model_invalid_tokenizer.tokenize_accept = false;
    model_invalid_tokenizer.tokenize_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::model_invalid);
    emel::text::conditioner::sm model_invalid_conditioner{};
    emel::embeddings::generator::sm model_invalid_generator{
      *fixture.model,
      model_invalid_conditioner,
      nullptr,
      emel::text::formatter::format_raw,
    };
    make_initialized_generator(
        model_invalid_tokenizer, nullptr, model_invalid_conditioner, model_invalid_generator);

    std::array<float, 1280> model_invalid_output = {};
    int32_t model_invalid_dimension = -1;
    emel::error::type model_invalid_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::embed_text model_invalid_request{
      messages,
      model_invalid_output,
      model_invalid_dimension,
    };
    model_invalid_request.error_out = &model_invalid_error;
    CHECK_FALSE(model_invalid_generator.process_event(model_invalid_request));
    CHECK(model_invalid_error ==
          emel::error::cast(emel::embeddings::generator::error::model_invalid));
  }
}
