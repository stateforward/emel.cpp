#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "emel/kernel/cq/detail.hpp"
#include "emel/kernel/cq/sm.hpp"

namespace {
using emel::cact::loader::tensor_view;
using emel::kernel::cq::event::gemv_request;

std::vector<uint8_t> make_blob(const std::vector<uint32_t> &indices,
                               const uint32_t out, const uint32_t in,
                               const uint32_t group, const uint32_t bits,
                               const std::vector<uint16_t> &norms) {
  const uint32_t in_pad = (in + group - 1u) / group * group;
  const size_t row_bytes = emel::kernel::cq::detail::packed_row_bytes(in_pad, bits);
  const size_t norm_count = static_cast<size_t>(out) * in_pad / group;
  std::vector<uint8_t> blob(static_cast<size_t>(out) * row_bytes + norm_count * 2u);
  for (uint32_t row = 0u; row < out; ++row) {
    uint8_t *packed = blob.data() + static_cast<size_t>(row) * row_bytes;
    for (uint32_t i = 0u; i < in_pad; ++i) {
      const uint32_t index = indices[static_cast<size_t>(row) * in_pad + i];
      if (bits == 5u) {
        const uint32_t encoded = index == 0u ? 3u : index - 1u;
        packed[i >> 2u] |= static_cast<uint8_t>(encoded << ((i & 3u) * 2u));
      } else {
        const size_t bit = static_cast<size_t>(i) * bits;
        packed[bit >> 3u] |= static_cast<uint8_t>(index << (bit & 7u));
        if ((bit & 7u) + bits > 8u)
          packed[(bit >> 3u) + 1u] |= static_cast<uint8_t>(index >> (8u - (bit & 7u)));
      }
    }
  }
  const size_t packed_bytes = static_cast<size_t>(out) * row_bytes;
  for (size_t i = 0u; i < norm_count; ++i) {
    blob[packed_bytes + i * 2u] = static_cast<uint8_t>(norms[i]);
    blob[packed_bytes + i * 2u + 1u] = static_cast<uint8_t>(norms[i] >> 8u);
  }
  return blob;
}

tensor_view make_view(const std::vector<uint8_t> &blob, const uint32_t out,
                      const uint32_t in, const uint32_t group,
                      const uint32_t bits) {
  return tensor_view{.dtype = 3u, .ndim = 2u, .shape = {out, in, 0u, 0u},
                     .nbytes = blob.size(), .group = group, .bits = bits,
                     .data = blob.data()};
}

} // namespace

TEST_CASE("CQ scalar unpack is LSB-first and applies normalized FWHT") {
  std::array<float, 28u> codebook{};
  codebook[0] = -0.5f; codebook[1] = -0.1f; codebook[2] = 0.1f; codebook[3] = 0.5f;
  const std::vector<uint32_t> indices{0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
  const auto blob = make_blob(indices, 1u, 8u, 8u, 2u, {0x3C00u});
  const auto view = make_view(blob, 1u, 8u, 8u, 2u);
  const std::array<float, 8u> activation{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::array<float, 8u> workspace{};
  std::array<float, 1u> output{};
  gemv_request request{view, codebook, activation, output, workspace};
  emel::kernel::cq::sm machine;
  emel::kernel::cq::event::dispatch_result result{};
  REQUIRE(machine.process_event(emel::kernel::cq::event::execute_scalar{request, result}));
  std::array<float, 8u> transformed = activation;
  emel::kernel::cq::detail::fwht(transformed.data(), 8u);
  float expected = 0.0f;
  for (uint32_t i = 0u; i < 8u; ++i) expected += codebook[indices[i]] * transformed[i];
  CHECK(output[0] == doctest::Approx(expected));
}

TEST_CASE("CQ ternary record uses crumb encoding and analytic centroid") {
  const std::vector<uint32_t> indices{0u, 1u, 2u, 0u, 2u, 1u, 0u, 2u};
  const auto blob = make_blob(indices, 1u, 8u, 8u, 5u, {0x3C00u});
  const auto view = make_view(blob, 1u, 8u, 8u, 5u);
  const std::array<float, 8u> activation{1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::array<float, 8u> workspace{};
  std::array<float, 1u> output{};
  std::array<float, 28u> codebook{};
  gemv_request request{view, codebook, activation, output, workspace};
  emel::kernel::cq::sm machine;
  emel::kernel::cq::event::dispatch_result result{};
  REQUIRE(machine.process_event(emel::kernel::cq::event::execute_scalar{request, result}));
  std::array<float, 8u> transformed = activation;
  emel::kernel::cq::detail::fwht(transformed.data(), 8u);
  float expected = 0.0f;
  for (uint32_t i = 0u; i < 8u; ++i)
    expected += emel::kernel::cq::detail::ternary_code(indices[i], 8u) * transformed[i];
  CHECK(output[0] == doctest::Approx(expected));
}

TEST_CASE("CQ3 and CQ4 preserve packed index order across byte boundaries") {
  for (const uint32_t bits : {3u, 4u}) {
    constexpr uint32_t in = 16u;
    constexpr uint32_t group = 8u;
    std::array<float, 28u> codebook{};
    const uint32_t offset = bits == 3u ? 4u : 12u;
    const uint32_t levels = 1u << bits;
    for (uint32_t i = 0u; i < levels; ++i) codebook[offset + i] = static_cast<float>(i + 1u) / 10.0f;
    std::vector<uint32_t> indices;
    for (uint32_t i = 0u; i < in * 2u; ++i) indices.push_back(i % levels);
    const auto blob = make_blob(indices, 2u, in, group, bits,
                                 {0x3C00u, 0x4000u, 0x3C00u, 0x4000u});
    const auto view = make_view(blob, 2u, in, group, bits);
    std::array<float, in> activation{};
    for (uint32_t i = 0u; i < in; ++i) activation[i] = static_cast<float>(i + 1u) / 8.0f;
    std::array<float, in> workspace{};
    std::array<float, 2u> scalar_output{};
    gemv_request request{view, codebook, activation, scalar_output, workspace};
    emel::kernel::cq::sm machine;
    emel::kernel::cq::event::dispatch_result result{};
    REQUIRE(machine.process_event(emel::kernel::cq::event::execute_scalar{request, result}));
    if (emel::kernel::cq::guard::avx2_supported(request)) {
      std::array<float, 2u> avx_output{};
      gemv_request avx_request{view, codebook, activation, avx_output, workspace};
      emel::kernel::cq::sm avx_machine;
      emel::kernel::cq::event::dispatch_result avx_result{};
      REQUIRE(avx_machine.process_event(emel::kernel::cq::event::execute_avx2{avx_request, avx_result}));
      CHECK(avx_output[0] == doctest::Approx(scalar_output[0]));
      CHECK(avx_output[1] == doctest::Approx(scalar_output[1]));
    }
  }
}

TEST_CASE("CQ guard rejects incomplete padded workspace and exposes ready state") {
  const std::vector<uint32_t> indices(16u, 0u);
  const auto blob = make_blob(indices, 1u, 13u, 8u, 4u, {0x3C00u, 0x3C00u});
  const auto view = make_view(blob, 1u, 13u, 8u, 4u);
  std::array<float, 28u> codebook{};
  std::array<float, 13u> activation{};
  std::array<float, 1u> output{};
  std::array<float, 13u> workspace{};
  gemv_request request{view, codebook, activation, output, workspace};
  emel::kernel::cq::sm machine;
  emel::kernel::cq::event::dispatch_result result{};
  CHECK_FALSE(machine.process_event(emel::kernel::cq::event::execute_scalar{request, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
}
