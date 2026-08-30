#include <array>
#include <cstdint>
#include <cstring>
#include <doctest/doctest.h>

#include "emel/kernel/hadamard/sm.hpp"

namespace {

using emel::kernel::hadamard::event::dispatch_result;

std::array<uint8_t, 8> pack_f16(const std::array<uint16_t, 4> &bits) {
  std::array<uint8_t, 8> bytes{};
  std::memcpy(bytes.data(), bits.data(), bytes.size());
  return bytes;
}

} // namespace

TEST_CASE("hadamard mlp row matches the reference d1/H/silu(d2 .)/H/d3 chain") {
  // d_model=3, hada_n=4; d1=[0.5,1,-1,2], d2=[1,0.5,0.25,-0.5],
  // d3=[2,1,0.5,1] as fp16; input [1,2,-1], skip [0.1,0.2,0.3].
  const auto d1 = pack_f16({0x3800u, 0x3c00u, 0xbc00u, 0x4000u});
  const auto d2 = pack_f16({0x3c00u, 0x3800u, 0x3400u, 0xb800u});
  const auto d3 = pack_f16({0x4000u, 0x3c00u, 0x3800u, 0x3c00u});
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
