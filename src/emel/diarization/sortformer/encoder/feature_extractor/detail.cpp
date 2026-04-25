#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace emel::diarization::sortformer::encoder::feature_extractor::detail {

namespace {

constexpr std::string_view k_filter_bank_name = "prep.feat.fb";
constexpr std::string_view k_window_name = "prep.feat.win";
constexpr float k_pi = 3.14159265358979323846f;

const emel::model::data::tensor_record * find_tensor(
    const emel::model::data & model_data,
    const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto & tensor = model_data.tensors[index];
    if (emel::model::tensor_name_view(model_data, tensor) == name) {
      return &tensor;
    }
  }

  return nullptr;
}

bool tensor_has_shape(const emel::model::data::tensor_record & tensor,
                      const int32_t n_dims,
                      const std::array<int64_t, 4> & dims) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims != n_dims) {
    return false;
  }

  for (int32_t dim = 0; dim < n_dims; ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] != dims[static_cast<size_t>(dim)]) {
      return false;
    }
  }

  return true;
}

float preemphasized_sample(std::span<const float> pcm, const int32_t index) noexcept {
  if (index < 0 || index >= static_cast<int32_t>(pcm.size())) {
    return 0.0f;
  }

  const float current = pcm[static_cast<size_t>(index)];
  if (index == 0) {
    return current;
  }

  return current - (k_preemphasis * pcm[static_cast<size_t>(index - 1)]);
}

void load_fft_frame(std::span<const float> pcm,
                    const int32_t frame,
                    std::span<const float, k_window_length> window,
                    std::array<float, k_fft_size> & real,
                    std::array<float, k_fft_size> & imag) noexcept {
  const int32_t frame_start = frame * k_hop_length;

  for (int32_t sample = 0; sample < k_fft_size; ++sample) {
    const int32_t pcm_index = frame_start + sample - (k_fft_size / 2);
    float value = preemphasized_sample(pcm, pcm_index);
    if (sample >= k_window_fft_padding &&
        sample < k_window_fft_padding + k_window_length) {
      value *= window[static_cast<size_t>(sample - k_window_fft_padding)];
    } else {
      value = 0.0f;
    }

    real[static_cast<size_t>(sample)] = value;
    imag[static_cast<size_t>(sample)] = 0.0f;
  }
}

void fft_in_place(std::array<float, k_fft_size> & real,
                  std::array<float, k_fft_size> & imag) noexcept {
  size_t j = 0u;
  for (size_t i = 1u; i < real.size(); ++i) {
    size_t bit = real.size() >> 1u;
    while ((j & bit) != 0u) {
      j ^= bit;
      bit >>= 1u;
    }
    j ^= bit;
    if (i < j) {
      std::swap(real[i], real[j]);
      std::swap(imag[i], imag[j]);
    }
  }

  for (size_t len = 2u; len <= real.size(); len <<= 1u) {
    const float angle = (-2.0f * k_pi) / static_cast<float>(len);
    const float w_len_real = std::cos(angle);
    const float w_len_imag = std::sin(angle);

    for (size_t offset = 0u; offset < real.size(); offset += len) {
      float w_real = 1.0f;
      float w_imag = 0.0f;
      const size_t half = len >> 1u;

      for (size_t index = 0u; index < half; ++index) {
        const size_t even = offset + index;
        const size_t odd = even + half;

        const float odd_real = (real[odd] * w_real) - (imag[odd] * w_imag);
        const float odd_imag = (real[odd] * w_imag) + (imag[odd] * w_real);

        real[odd] = real[even] - odd_real;
        imag[odd] = imag[even] - odd_imag;
        real[even] += odd_real;
        imag[even] += odd_imag;

        const float next_w_real = (w_real * w_len_real) - (w_imag * w_len_imag);
        const float next_w_imag = (w_real * w_len_imag) + (w_imag * w_len_real);
        w_real = next_w_real;
        w_imag = next_w_imag;
      }
    }
  }
}

void compute_power_spectrum(const std::array<float, k_fft_size> & real,
                            const std::array<float, k_fft_size> & imag,
                            std::array<float, k_fft_bin_count> & power) noexcept {
  for (int32_t bin = 0; bin < k_fft_bin_count; ++bin) {
    const float real_value = real[static_cast<size_t>(bin)];
    const float imag_value = imag[static_cast<size_t>(bin)];
    power[static_cast<size_t>(bin)] =
        (real_value * real_value) + (imag_value * imag_value);
  }
}

void compute_mel_features(std::span<const float, k_fft_bin_count * k_feature_bin_count> filter_bank,
                          const std::array<float, k_fft_bin_count> & power,
                          std::span<float, k_feature_bin_count> feature_row) noexcept {
  for (int32_t mel = 0; mel < k_feature_bin_count; ++mel) {
    float acc = 0.0f;
    const size_t row_base = static_cast<size_t>(mel) * static_cast<size_t>(k_fft_bin_count);

    for (int32_t bin = 0; bin < k_fft_bin_count; ++bin) {
      acc += filter_bank[row_base + static_cast<size_t>(bin)] *
          power[static_cast<size_t>(bin)];
    }

    feature_row[static_cast<size_t>(mel)] = std::log(acc + k_log_zero_guard_value);
  }
}

}  // namespace

contract make_contract(const emel::model::data & model_data) noexcept {
  contract next = {};
  next.filter_bank.tensor = find_tensor(model_data, k_filter_bank_name);
  next.filter_bank.name = k_filter_bank_name;
  next.window.tensor = find_tensor(model_data, k_window_name);
  next.window.name = k_window_name;
  return next;
}

bool contract_valid(const contract & feature_contract) noexcept {
  return feature_contract.filter_bank.tensor != nullptr &&
      tensor_has_shape(*feature_contract.filter_bank.tensor,
                       3,
                       {k_fft_bin_count, k_feature_bin_count, 1, 0}) &&
      feature_contract.window.tensor != nullptr &&
      tensor_has_shape(*feature_contract.window.tensor,
                       1,
                       {k_window_length, 0, 0, 0});
}

void compute(std::span<const float> pcm,
             const contract & feature_contract,
             std::span<float> features) noexcept {
  const auto filter_bank =
      tensor_data<k_fft_bin_count * k_feature_bin_count>(*feature_contract.filter_bank.tensor);
  const auto window = tensor_data<k_window_length>(*feature_contract.window.tensor);

  std::array<float, k_fft_size> real = {};
  std::array<float, k_fft_size> imag = {};
  std::array<float, k_fft_bin_count> power = {};

  for (int32_t frame = 0; frame < k_feature_frame_count; ++frame) {
    const size_t feature_offset =
        static_cast<size_t>(frame) * static_cast<size_t>(k_feature_bin_count);
    auto feature_row = std::span<float, k_feature_bin_count>{
        features.data() + feature_offset,
        static_cast<size_t>(k_feature_bin_count)};

    load_fft_frame(pcm, frame, window, real, imag);
    fft_in_place(real, imag);
    compute_power_spectrum(real, imag, power);
    compute_mel_features(filter_bank, power, feature_row);
  }
}

}  // namespace emel::diarization::sortformer::encoder::feature_extractor::detail
