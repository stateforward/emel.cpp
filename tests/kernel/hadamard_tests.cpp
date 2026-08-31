#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <doctest/doctest.h>
#include "emel/kernel/hadamard/sm.hpp"

namespace {

using emel::kernel::hadamard::event::dispatch_result;

std::array<uint8_t, 8> pack_f16_bits(const std::array<uint16_t, 4> &bits) {
  std::array<uint8_t, 8> bytes{};
  std::memcpy(bytes.data(), bits.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> pack_f16_values(const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(uint16_t));
  for (size_t i = 0u; i < values.size(); ++i) {
    const uint16_t bits = emel::kernel::detail::quant::fp32_to_fp16(values[i]);
    std::memcpy(bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
  }
  return bytes;
}
bool host_has_hadamard_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return emel::kernel::x86_64::detail::detect_avx2() &&
         emel::kernel::x86_64::detail::detect_fma() &&
         emel::kernel::x86_64::detail::detect_f16c();
#else
  return false;
#endif
}

} // namespace

TEST_CASE("hadamard mlp row matches the reference d1/H/silu(d2 .)/H/d3 chain") {
  // d_model=3, hada_n=4; d1=[0.5,1,-1,2], d2=[1,0.5,0.25,-0.5],
  // d3=[2,1,0.5,1] as fp16; input [1,2,-1], skip [0.1,0.2,0.3].
  const auto d1 = pack_f16_bits({0x3800u, 0x3c00u, 0xbc00u, 0x4000u});
  const auto d2 = pack_f16_bits({0x3c00u, 0x3800u, 0x3400u, 0xb800u});
  const auto d3 = pack_f16_bits({0x4000u, 0x3c00u, 0x3800u, 0x3c00u});
  const std::array<float, 3> input{1.0f, 2.0f, -1.0f};
  const std::array<float, 3> skip{0.1f, 0.2f, 0.3f};
  std::array<float, 4> workspace{};
  std::array<float, 3> output{};
  const emel::kernel::hadamard::event::mlp_row_request request{
      .input = input,
      .skip = skip,
      .d1 = d1,
      .d2 = d2,
      .d3 = d3,
      .d_model = 3u,
      .hada_n = 4u,
      .workspace = workspace,
      .output = output};
  emel::kernel::hadamard::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{request, result}));
  CHECK(output[0] == doctest::Approx(2.041928770519778f));
  CHECK(output[1] == doctest::Approx(0.8224664254070599f));
  CHECK(output[2] == doctest::Approx(0.530677107221645f));
}

TEST_CASE("hadamard guard rejects non-power-of-two hada_n") {
  const std::array<uint8_t, 8> d{};
  const std::array<float, 3> input{};
  const std::array<float, 3> skip{};
  std::array<float, 4> workspace{};
  std::array<float, 3> output{};
  const emel::kernel::hadamard::event::mlp_row_request request{
      .input = input,
      .skip = skip,
      .d1 = d,
      .d2 = d,
      .d3 = d,
      .d_model = 3u,
      .hada_n = 3u,
      .workspace = workspace,
      .output = output};
  emel::kernel::hadamard::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{request, result}));
  CHECK(machine.is(
      stateforward::sml::state<emel::kernel::hadamard::state_ready>));
}

TEST_CASE("hadamard AVX2 512 route matches scalar arbitrary fp16 diagonals") {
  if (!host_has_hadamard_avx2())
    return;
  constexpr uint32_t n = 512u;
  std::vector<float> input(n);
  std::vector<float> skip(n);
  std::vector<float> d1_values(n);
  std::vector<float> d2_values(n);
  std::vector<float> d3_values(n);
  for (uint32_t i = 0u; i < n; ++i) {
    input[i] = static_cast<float>(static_cast<int32_t>((i * 37u) % 101u) - 50) *
               0.03125f;
    skip[i] = static_cast<float>(static_cast<int32_t>((i * 19u) % 83u) - 41) *
              0.015625f;
    d1_values[i] =
        static_cast<float>(static_cast<int32_t>((i * 13u) % 29u) - 14) *
        0.125f;
    d2_values[i] =
        static_cast<float>(static_cast<int32_t>((i * 17u) % 31u) - 15) *
        0.09375f;
    d3_values[i] =
        static_cast<float>(static_cast<int32_t>((i * 23u) % 37u) - 18) *
        0.0625f;
  }
  const auto d1 = pack_f16_values(d1_values);
  const auto d2 = pack_f16_values(d2_values);
  const auto d3 = pack_f16_values(d3_values);
  std::vector<float> scalar_workspace(n);
  std::vector<float> avx2_workspace(n);
  std::vector<float> scalar_output(n);
  std::vector<float> avx2_output(n);
  const emel::kernel::hadamard::event::mlp_row_request scalar_request{
      input, skip, d1, d2, d3, n, n, scalar_workspace, scalar_output};
  const emel::kernel::hadamard::event::mlp_row_request avx2_request{
      input, skip, d1, d2, d3, n, n, avx2_workspace, avx2_output};
  emel::kernel::hadamard::sm machine;
  dispatch_result scalar_result{};
  dispatch_result avx2_result{};
  REQUIRE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{scalar_request,
                                                      scalar_result}));
  REQUIRE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row_avx2{avx2_request,
                                                           avx2_result}));
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (uint32_t i = 0u; i < n; ++i) {
    const float diff = std::abs(scalar_output[i] - avx2_output[i]);
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, diff / std::max(std::abs(scalar_output[i]), 1.0f));
  }
  MESSAGE("Hadamard AVX2 vs scalar max_abs=", max_abs,
          " max_rel=", max_rel);
  CHECK(max_abs <= 2.0e-5f);
  CHECK(max_rel <= 2.0e-5f);
}

TEST_CASE("hadamard scalar preserves zero-padding semantics") {
  constexpr uint32_t d_model = 300u;
  constexpr uint32_t n = 512u;
  std::vector<float> input(d_model);
  std::vector<float> skip(d_model);
  std::vector<float> diagonals(n, 1.0f);
  for (uint32_t i = 0u; i < d_model; ++i) {
    input[i] = static_cast<float>(static_cast<int32_t>(i % 17u) - 8) * 0.125f;
    skip[i] = static_cast<float>(static_cast<int32_t>(i % 11u) - 5) * 0.0625f;
  }
  const auto d = pack_f16_values(diagonals);
  std::vector<float> workspace(n, 99.0f);
  std::vector<float> output(d_model);
  const emel::kernel::hadamard::event::mlp_row_request request{
      input, skip, d, d, d, d_model, n, workspace, output};
  emel::kernel::hadamard::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{request, result}));
  for (const float value : output)
    CHECK(std::isfinite(value));
}

TEST_CASE("hadamard rejects writable aliases without writes") {
  constexpr uint32_t n = 512u;
  std::vector<float> input(n, 0.25f);
  std::vector<float> skip(n, -0.5f);
  std::vector<float> diagonals(n, 1.0f);
  const auto d = pack_f16_values(diagonals);
  std::vector<float> storage(n * 2u, 7.0f);
  std::vector<float> output(n, 11.0f);
  const auto storage_before = storage;
  const auto output_before = output;
  emel::kernel::hadamard::sm machine;
  dispatch_result result{};
  SUBCASE("output aliases workspace") {
    const emel::kernel::hadamard::event::mlp_row_request request{
        input, skip, d, d, d, n, n, std::span<float>{storage}.first(n),
        std::span<float>{storage}.first(n)};
    CHECK_FALSE(machine.process_event(
        emel::kernel::hadamard::event::execute_mlp_row_avx2{request, result}));
  }
  SUBCASE("workspace aliases input") {
    const emel::kernel::hadamard::event::mlp_row_request request{
        std::span<const float>{storage}.first(n), skip, d, d, d, n, n,
        std::span<float>{storage}.first(n), output};
    CHECK_FALSE(machine.process_event(
        emel::kernel::hadamard::event::execute_mlp_row{request, result}));
  }
  CHECK(storage == storage_before);
  CHECK(output == output_before);
}

TEST_CASE("hadamard rejects every writable and read-only alias without writes") {
  constexpr uint32_t n = 512u;
  std::vector<float> float_storage(n * 4u, 0.25f);
  std::vector<uint8_t> byte_storage(n * sizeof(uint16_t), 0u);
  std::vector<float> separate(n, -0.5f);
  const auto float_before = float_storage;
  const auto byte_before = byte_storage;
  const auto separate_before = separate;
  const auto d = std::span<const uint8_t>{byte_storage};
  auto rejects = [&](const std::span<const float> input,
                     const std::span<const float> skip,
                     const std::span<const uint8_t> d1,
                     const std::span<const uint8_t> d2,
                     const std::span<const uint8_t> d3,
                     const std::span<float> workspace,
                     const std::span<float> output) {
    const emel::kernel::hadamard::event::mlp_row_request request{
        input, skip, d1, d2, d3, n, n, workspace, output};
    emel::kernel::hadamard::sm machine;
    dispatch_result result{};
    CHECK_FALSE(machine.process_event(
        emel::kernel::hadamard::event::execute_mlp_row_avx2{request, result}));
  };
  const auto floats = std::span<float>{float_storage};
  const auto input = std::span<const float>{floats.subspan(0u, n)};
  const auto skip = std::span<const float>{floats.subspan(n, n)};
  const auto workspace = floats.subspan(n * 2u, n);
  const auto output = floats.subspan(n * 3u, n);
  SUBCASE("workspace aliases skip") {
    rejects(input, workspace, d, d, d, workspace, output);
  }
  SUBCASE("workspace aliases d1") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(workspace.data()), n * 2u};
    rejects(input, skip, bytes, d, d, workspace, output);
  }
  SUBCASE("workspace aliases d2") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(workspace.data()), n * 2u};
    rejects(input, skip, d, bytes, d, workspace, output);
  }
  SUBCASE("workspace aliases d3") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(workspace.data()), n * 2u};
    rejects(input, skip, d, d, bytes, workspace, output);
  }
  SUBCASE("output aliases input") {
    rejects(input, skip, d, d, d, workspace, floats.subspan(0u, n));
  }
  SUBCASE("output aliases skip") {
    rejects(input, skip, d, d, d, workspace, floats.subspan(n, n));
  }
  SUBCASE("output aliases d1") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(output.data()), n * 2u};
    rejects(input, skip, bytes, d, d, workspace, output);
  }
  SUBCASE("output aliases d2") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(output.data()), n * 2u};
    rejects(input, skip, d, bytes, d, workspace, output);
  }
  SUBCASE("output aliases d3") {
    const auto bytes = std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(output.data()), n * 2u};
    rejects(input, skip, d, d, bytes, workspace, output);
  }
  CHECK(float_storage == float_before);
  CHECK(byte_storage == byte_before);
  CHECK(separate == separate_before);
}

TEST_CASE("hadamard AVX2 invalid geometry rejects while scalar fallback accepts") {
  constexpr uint32_t d_model = 256u;
  constexpr uint32_t n = 512u;
  std::vector<float> input(d_model, 0.25f);
  std::vector<float> skip(d_model, -0.5f);
  std::vector<float> diagonals(n, 1.0f);
  const auto d = pack_f16_values(diagonals);
  std::vector<float> workspace(n);
  std::vector<float> output(d_model);
  const emel::kernel::hadamard::event::mlp_row_request request{
      input, skip, d, d, d, d_model, n, workspace, output};
  emel::kernel::hadamard::sm machine;
  dispatch_result avx2_result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row_avx2{request,
                                                           avx2_result}));
  dispatch_result scalar_result{};
  CHECK(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{request,
                                                      scalar_result}));
}

TEST_CASE("hadamard accepts byte-aligned fp16 diagonal payloads") {
  constexpr uint32_t d_model = 3u;
  constexpr uint32_t n = 4u;
  std::array<float, d_model> input{1.0f, 2.0f, -1.0f};
  std::array<float, d_model> skip{0.1f, 0.2f, 0.3f};
  const auto packed = pack_f16_bits({0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u});
  std::array<uint8_t, packed.size() + 1u> unaligned{};
  std::copy(packed.begin(), packed.end(), unaligned.begin() + 1u);
  const auto d = std::span<const uint8_t>{unaligned}.subspan(1u, packed.size());
  std::array<float, n> workspace{};
  std::array<float, d_model> output{};
  const emel::kernel::hadamard::event::mlp_row_request request{
      input, skip, d, d, d, d_model, n, workspace, output};
  emel::kernel::hadamard::sm machine;
  dispatch_result result{};
  CHECK(machine.process_event(
      emel::kernel::hadamard::event::execute_mlp_row{request, result}));
  for (const float value : output)
    CHECK(std::isfinite(value));
}
