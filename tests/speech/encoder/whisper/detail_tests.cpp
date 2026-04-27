#include "emel/speech/encoder/whisper/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "doctest/doctest.h"

namespace {

using tensor_record = emel::model::data::tensor_record;

template <typename block_type, size_t block_count>
tensor_record make_tensor(std::array<block_type, block_count> & blocks,
                          const int32_t type,
                          const int64_t cols,
                          const int64_t rows) {
  tensor_record tensor{};
  tensor.type = type;
  tensor.n_dims = 2;
  tensor.dims[0] = cols;
  tensor.dims[1] = rows;
  tensor.data = blocks.data();
  tensor.data_size = sizeof(block_type) * block_count;
  return tensor;
}

}  // namespace

TEST_CASE("whisper detail tensor shape validates data and dimensions") {
  tensor_record tensor{};
  int32_t payload = 7;
  tensor.data = &payload;
  tensor.data_size = sizeof(payload);
  tensor.n_dims = 2;
  tensor.dims[0] = 32;
  tensor.dims[1] = 2;

  CHECK(emel::speech::encoder::whisper::detail::tensor_has_shape(tensor, 2, {32, 2, 0, 0}));
  CHECK_FALSE(emel::speech::encoder::whisper::detail::tensor_has_shape(tensor, 2, {16, 2, 0, 0}));

  tensor.n_dims = 1;
  CHECK_FALSE(emel::speech::encoder::whisper::detail::tensor_has_shape(tensor, 2, {32, 2, 0, 0}));
  tensor.n_dims = 2;

  tensor.data_size = 0;
  CHECK_FALSE(emel::speech::encoder::whisper::detail::tensor_has_shape(tensor, 2, {32, 2, 0, 0}));
  tensor.data_size = sizeof(payload);

  tensor.data = nullptr;
  CHECK_FALSE(emel::speech::encoder::whisper::detail::tensor_has_shape(tensor, 2, {32, 2, 0, 0}));
}

TEST_CASE("whisper detail finds tensors and normalizes softmax values") {
  auto model = std::make_unique<emel::model::data>();
  const char name[] = "target";
  std::copy_n(name, sizeof(name) - 1u, model->name_storage.begin());
  model->n_tensors = 1;
  model->tensors[0].name_offset = 0;
  model->tensors[0].name_length = sizeof(name) - 1u;

  CHECK(emel::speech::encoder::whisper::detail::find_tensor(*model, "target") == &model->tensors[0]);
  CHECK(emel::speech::encoder::whisper::detail::find_tensor(*model, "missing") == nullptr);

  std::array<float, 3> values{1.0f, 3.0f, 2.0f};
  emel::speech::encoder::whisper::detail::softmax(values.data(), values.size());
  CHECK(values[1] > values[2]);
  CHECK(values[2] > values[0]);
  CHECK(values[0] + values[1] + values[2] == doctest::Approx(1.0f));

  std::array<float, 1> single{42.0f};
  emel::speech::encoder::whisper::detail::softmax(single.data(), single.size());
  CHECK(single[0] == doctest::Approx(1.0f));

  CHECK(emel::speech::encoder::whisper::detail::mel_frame_count_for_samples(16'000u) == 100u);
  CHECK(emel::speech::encoder::whisper::detail::mel_frame_count_for_samples(1'000'000u) ==
        emel::speech::encoder::whisper::detail::k_max_mel_frame_count);
}

TEST_CASE("whisper detail q4 helpers decode both lanes and dot rows") {
  namespace kernel = emel::kernel::detail;
  namespace whisper = emel::speech::encoder::whisper::detail;

  std::array<kernel::quant::block_q4_0, 2> q4_0_rows{};
  std::array<kernel::quant::block_q4_1, 2> q4_1_rows{};
  for (size_t row = 0; row < q4_0_rows.size(); ++row) {
    q4_0_rows[row].d = kernel::quant::fp32_to_fp16(0.5f + static_cast<float>(row));
    q4_1_rows[row].d = kernel::quant::fp32_to_fp16(0.25f + static_cast<float>(row));
    q4_1_rows[row].m = kernel::quant::fp32_to_fp16(-1.0f + static_cast<float>(row));
    for (size_t lane = 0; lane < kernel::quant::QK4_0 / 2u; ++lane) {
      const uint8_t low = static_cast<uint8_t>(lane & 0x0fu);
      const uint8_t high = static_cast<uint8_t>(15u - low);
      q4_0_rows[row].qs[lane] = static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4u));
      q4_1_rows[row].qs[lane] = static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4u));
    }
  }

  auto q4_0_tensor = make_tensor(q4_0_rows,
                                 kernel::dtype_q4_0,
                                 static_cast<int64_t>(kernel::quant::QK4_0),
                                 2);
  auto q4_1_tensor = make_tensor(q4_1_rows,
                                 kernel::dtype_q4_1,
                                 static_cast<int64_t>(kernel::quant::QK4_1),
                                 2);

  CHECK(whisper::read_matrix_q4_0_value(q4_0_tensor, 0, 0) == doctest::Approx(-4.0f));
  CHECK(whisper::read_matrix_q4_0_value(q4_0_tensor, 0, 16) == doctest::Approx(3.5f));
  CHECK(whisper::read_matrix_q4_1_value(q4_1_tensor, 0, 0) == doctest::Approx(-1.0f));
  CHECK(whisper::read_matrix_q4_1_value(q4_1_tensor, 0, 16) == doctest::Approx(2.75f));

  std::array<float, kernel::quant::QK4_0> input{};
  input.fill(1.0f);

  const float q4_0_dot = whisper::dot_linear_row<whisper::linear_weight_variant::q4_0>(
      q4_0_tensor, 0, input.data(), input.size());
  const float q4_1_dot = whisper::dot_linear_row<whisper::linear_weight_variant::q4_1>(
      q4_1_tensor, 0, input.data(), input.size());

  CHECK(q4_0_dot == doctest::Approx(-8.0f));
  CHECK(q4_1_dot == doctest::Approx(28.0f));

  std::array<kernel::quant::block_q8_0, 2> bias_blocks{};
  bias_blocks[0].d = kernel::quant::fp32_to_fp16(1.0f);
  bias_blocks[0].qs[0] = 2;
  bias_blocks[0].qs[1] = -3;
  auto bias_tensor = make_tensor(bias_blocks,
                                 kernel::dtype_q8_0,
                                 static_cast<int64_t>(kernel::quant::QK8_0),
                                 2);
  std::array<float, 2> linear_out{};
  whisper::linear<whisper::linear_weight_variant::q4_0,
                  kernel::quant::QK4_0,
                  2>(q4_0_tensor, bias_tensor, input.data(), linear_out.data());
  CHECK(linear_out[0] == doctest::Approx(-6.0f));

  whisper::linear_no_bias<whisper::linear_weight_variant::q4_1,
                          kernel::quant::QK4_1,
                          2>(q4_1_tensor, input.data(), linear_out.data());
  CHECK(linear_out[0] == doctest::Approx(28.0f));
}

TEST_CASE("whisper detail q8 neon row helper handles empty and populated rows") {
#if defined(__ARM_NEON) && defined(__aarch64__)
  namespace kernel = emel::kernel::detail;
  namespace whisper = emel::speech::encoder::whisper::detail;

  std::array<kernel::quant::block_q8_0, 1> blocks{};
  blocks[0].d = kernel::quant::fp32_to_fp16(0.5f);
  std::array<float, kernel::quant::QK8_0> input{};
  for (size_t lane = 0; lane < input.size(); ++lane) {
    blocks[0].qs[lane] = static_cast<int8_t>(lane % 4u);
    input[lane] = 1.0f;
  }

  CHECK(whisper::dot_q8_0_row_neon(blocks.data(), input.data(), 0u) == doctest::Approx(0.0f));
  CHECK(whisper::dot_q8_0_row_neon(blocks.data(), input.data(), 1u) == doctest::Approx(24.0f));
#else
  CHECK(true);
#endif
}

TEST_CASE("whisper detail q8 helpers decode rows and linear outputs") {
  namespace kernel = emel::kernel::detail;
  namespace whisper = emel::speech::encoder::whisper::detail;

  std::array<kernel::quant::block_q8_0, 4> rows{};
  for (size_t row = 0; row < 2u; ++row) {
    rows[row].d = kernel::quant::fp32_to_fp16(0.25f + static_cast<float>(row));
    for (size_t lane = 0; lane < kernel::quant::QK8_0; ++lane) {
      rows[row].qs[lane] = static_cast<int8_t>((lane % 5u) - 2u);
    }
  }

  auto weight = make_tensor(rows,
                            kernel::dtype_q8_0,
                            static_cast<int64_t>(kernel::quant::QK8_0),
                            2);
  CHECK(whisper::read_q8_0_value(weight, 0) == doctest::Approx(-0.5f));
  CHECK(whisper::read_q8_0_value(weight, kernel::quant::QK8_0) == doctest::Approx(-2.5f));

  std::array<float, kernel::quant::QK8_0> input{};
  input.fill(1.0f);
  const float dot = whisper::dot_linear_row<whisper::linear_weight_variant::q8_0>(
      weight, 0, input.data(), input.size());
  CHECK(dot == doctest::Approx(-0.75f));

  std::array<kernel::quant::block_q8_0, 1> bias_blocks{};
  bias_blocks[0].d = kernel::quant::fp32_to_fp16(0.5f);
  bias_blocks[0].qs[0] = 3;
  bias_blocks[0].qs[1] = -4;
  auto bias = make_tensor(bias_blocks,
                          kernel::dtype_q8_0,
                          static_cast<int64_t>(kernel::quant::QK8_0),
                          1);

  std::array<float, 2> output{};
  whisper::linear<whisper::linear_weight_variant::q8_0,
                  kernel::quant::QK8_0,
                  2>(weight, bias, input.data(), output.data());
  CHECK(output[0] == doctest::Approx(0.75f).epsilon(0.001));

  whisper::linear_no_bias<whisper::linear_weight_variant::q8_0,
                          kernel::quant::QK8_0,
                          2>(weight, input.data(), output.data());
  CHECK(output[1] == doctest::Approx(-3.75f).epsilon(0.001));
}

TEST_CASE("whisper detail convolution helpers cover interior frame paths") {
  namespace kernel = emel::kernel::detail;
  namespace whisper = emel::speech::encoder::whisper::detail;

  constexpr uint64_t mel_frames = 3u;
  constexpr uint64_t encoder_frames = 2u;
  const uint64_t conv1_weight_count =
      3u * static_cast<uint64_t>(whisper::k_mel_bin_count) *
      static_cast<uint64_t>(whisper::k_embedding_length);
  std::vector<uint16_t> conv1_weights(static_cast<size_t>(conv1_weight_count));
  std::vector<float> mel(static_cast<size_t>(
      static_cast<uint64_t>(whisper::k_mel_bin_count) * mel_frames));
  std::vector<float> conv1(static_cast<size_t>(
      static_cast<uint64_t>(whisper::k_embedding_length) * mel_frames));

  for (uint16_t &weight : conv1_weights) {
    weight = kernel::quant::fp32_to_fp16(0.0f);
  }
  std::array<kernel::quant::block_q8_0, 12> bias_blocks{};
  for (auto &block : bias_blocks) {
    block.d = kernel::quant::fp32_to_fp16(1.0f);
  }

  tensor_record conv1_weight{};
  conv1_weight.data = conv1_weights.data();
  conv1_weight.data_size = conv1_weights.size() * sizeof(uint16_t);
  conv1_weight.n_dims = 3;
  conv1_weight.dims[1] = whisper::k_mel_bin_count;
  tensor_record bias = make_tensor(
      bias_blocks, kernel::dtype_q8_0, whisper::k_embedding_length, 1);

  whisper::run_conv1<whisper::aux_weight_variant::q8_0>(
      mel.data(), mel_frames, conv1_weight, bias, conv1.data());

  const uint64_t conv2_weight_count =
      3u * static_cast<uint64_t>(whisper::k_embedding_length) *
      static_cast<uint64_t>(whisper::k_embedding_length);
  std::vector<uint16_t> conv2_weights(static_cast<size_t>(conv2_weight_count));
  std::vector<float> hidden(static_cast<size_t>(
      static_cast<uint64_t>(whisper::k_embedding_length) * encoder_frames));
  for (uint16_t &weight : conv2_weights) {
    weight = kernel::quant::fp32_to_fp16(0.0f);
  }

  tensor_record conv2_weight{};
  conv2_weight.data = conv2_weights.data();
  conv2_weight.data_size = conv2_weights.size() * sizeof(uint16_t);
  conv2_weight.n_dims = 3;
  conv2_weight.dims[1] = whisper::k_embedding_length;

  whisper::run_conv2<whisper::aux_weight_variant::q8_0>(
      conv1.data(), mel_frames, conv2_weight, bias, hidden.data());
  CHECK(std::all_of(hidden.begin(), hidden.end(),
                    [](const float value) { return std::isfinite(value); }));
}

TEST_CASE("whisper detail f32 auxiliary readers and timestamp blocking are covered") {
  namespace whisper = emel::speech::encoder::whisper::detail;

  std::array<float, 4> f32_values{1.25f, -2.5f, 3.75f, 4.5f};
  tensor_record f32_tensor{};
  f32_tensor.type = emel::kernel::detail::dtype_f32;
  f32_tensor.n_dims = 1;
  f32_tensor.dims[0] = static_cast<int64_t>(f32_values.size());
  f32_tensor.data = f32_values.data();
  f32_tensor.data_size = sizeof(float) * f32_values.size();

  CHECK(whisper::read_aux_vector<whisper::aux_weight_variant::f32>(f32_tensor, 1) ==
        doctest::Approx(-2.5f));
  CHECK(whisper::read_aux_matrix<whisper::aux_weight_variant::f32>(f32_tensor, 2) ==
        doctest::Approx(3.75f));

  std::vector<float> logits(static_cast<size_t>(whisper::k_vocab_size), -1000.0f);
  logits[42] = 100.0f;
  logits[static_cast<size_t>(whisper::k_token_timestamp_begin)] = 0.0f;
  const std::array<int32_t, 2> generated{42, whisper::k_token_timestamp_begin};
  const whisper::decode_policy_runtime policy{};
  float confidence = 0.0f;
  const int32_t token = whisper::select_greedy_timestamp_aware_token(
      policy, logits.data(), generated.data(), generated.size(), false,
      confidence);
  CHECK(token >= whisper::k_token_timestamp_begin);
}

TEST_CASE("whisper detail timestamp policy suppresses initial and control tokens") {
  namespace whisper = emel::speech::encoder::whisper::detail;

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

TEST_CASE("whisper detail spectral helpers prepare stable frame transforms") {
  namespace whisper = emel::speech::encoder::whisper::detail;

  std::array<float, 8> real{1.0f, 0.5f, -0.25f, 0.75f, 0.0f, -0.5f, 0.25f, -1.0f};
  std::array<float, 8> imag{};
  const auto original = real;

  whisper::fft_radix2(real.data(), imag.data(), real.size(), false);
  whisper::fft_radix2(real.data(), imag.data(), real.size(), true);
  for (size_t index = 0; index < real.size(); ++index) {
    CHECK(real[index] == doctest::Approx(original[index]).epsilon(0.0001));
    CHECK(imag[index] == doctest::Approx(0.0f).epsilon(0.0001));
  }

  std::array<float, whisper::k_fft_size> window{};
  std::array<float, whisper::k_fft_size> chirp_real{};
  std::array<float, whisper::k_fft_size> chirp_imag{};
  whisper::prepare_frame_tables(window.data(), chirp_real.data(), chirp_imag.data());
  CHECK(window[0] == doctest::Approx(0.0f));
  CHECK(window[whisper::k_fft_size / 2] == doctest::Approx(1.0f));
  CHECK(chirp_real[0] == doctest::Approx(1.0f));
  CHECK(chirp_imag[0] == doctest::Approx(0.0f));
}
