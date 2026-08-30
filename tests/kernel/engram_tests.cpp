#include <array>
#include <cstdint>
#include <cstring>
#include <doctest/doctest.h>

#include "emel/kernel/engram/sm.hpp"

namespace {

using emel::kernel::engram::event::dispatch_result;

} // namespace

TEST_CASE("engram hash rows reproduce the reference FNV-mix indices") {
  // orders=(2,3), heads=1, slots=64, tokens=[5,7,9] all valid; expectations
  // computed with the reference `engram_indices` arithmetic.
  const std::array<int32_t, 3> tokens{5, 7, 9};
  const std::array<uint8_t, 3> valid{1u, 1u, 1u};
  const std::array<uint32_t, 2> orders{2u, 3u};
  std::array<uint32_t, 6> indices{};
  std::array<float, 6> ngram_ok{};
  const emel::kernel::engram::event::hash_rows_request request{
      .tokens = tokens,
      .valid = valid,
      .positions = 3u,
      .orders = orders,
      .num_orders = 2u,
      .heads = 1u,
      .slots = 64u,
      .indices = indices,
      .ngram_ok = ngram_ok};
  emel::kernel::engram::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::engram::event::execute_hash_rows{request, result}));
  // Row-major (position, table): table 0 = order 2, table 1 = order 3.
  CHECK(indices[0] == 37u);
  CHECK(indices[1] == 82u - 64u);
  CHECK(indices[2] == 14u);
  CHECK(indices[3] == 97u - 64u);
  CHECK(indices[4] == 8u);
  CHECK(indices[5] == 104u - 64u);
  CHECK(ngram_ok[0] == 0.0f); // order-2 gram at position 0 incomplete
  CHECK(ngram_ok[1] == 0.0f);
  CHECK(ngram_ok[2] == 1.0f); // position 1, order 2
  CHECK(ngram_ok[3] == 0.0f);
  CHECK(ngram_ok[4] == 1.0f); // position 2, order 2
  CHECK(ngram_ok[5] == 1.0f); // position 2, order 3
}

TEST_CASE("engram conv taps accumulate dilated causal taps over value rows") {
  // positions=3, window=1, outputs=2, taps=2, dilation=1, dim=2;
  // taps fp16 [[1, 0.5], [0.25, 2]]; values [[1,2],[3,4],[5,6]].
  const std::array<float, 6> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<uint8_t, 3> valid{1u, 1u, 1u};
  const std::array<uint16_t, 4> tap_bits{0x3c00u, 0x3800u, 0x3400u, 0x4000u};
  std::array<uint8_t, 8> taps{};
  std::memcpy(taps.data(), tap_bits.data(), taps.size());
  std::array<float, 4> output{};
  const emel::kernel::engram::event::conv_taps_request request{.values = values,
                                                               .valid = valid,
                                                               .positions = 3u,
                                                               .window = 1u,
                                                               .outputs = 2u,
                                                               .conv_taps = 2u,
                                                               .dilation = 1u,
                                                               .taps = taps,
                                                               .dim = 2u,
                                                               .output =
                                                                   output};
  emel::kernel::engram::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::engram::event::execute_conv_taps{request, result}));
  CHECK(output[0] == doctest::Approx(3.25f));
  CHECK(output[1] == doctest::Approx(6.0f));
  CHECK(output[2] == doctest::Approx(5.75f));
  CHECK(output[3] == doctest::Approx(11.0f));
}

TEST_CASE("engram alpha gate blends value rows by the rms-unit dot sigmoid") {
  const std::array<float, 4> u{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> key{0.5f, -1.0f, 2.0f, 0.0f};
  const std::array<float, 4> value{1.0f, 1.0f, -1.0f, 0.5f};
  std::array<float, 4> output{};
  const emel::kernel::engram::event::alpha_gate_request request{
      .u = u, .key = key, .value = value, .dim = 4u, .output = output};
  emel::kernel::engram::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::engram::event::execute_alpha_gate{request, result}));
  CHECK(output[0] == doctest::Approx(1.6719763278961182f));
  CHECK(output[1] == doctest::Approx(2.671976327896118f));
  CHECK(output[2] == doctest::Approx(2.328023672103882f));
  CHECK(output[3] == doctest::Approx(4.3359880447387695f));
}

TEST_CASE("engram hash guard rejects zero slots") {
  const std::array<int32_t, 3> tokens{};
  const std::array<uint8_t, 3> valid{};
  const std::array<uint32_t, 2> orders{2u, 3u};
  std::array<uint32_t, 6> indices{};
  std::array<float, 6> ngram_ok{};
  const emel::kernel::engram::event::hash_rows_request request{
      .tokens = tokens,
      .valid = valid,
      .positions = 3u,
      .orders = orders,
      .num_orders = 2u,
      .heads = 1u,
      .slots = 0u,
      .indices = indices,
      .ngram_ok = ngram_ok};
  emel::kernel::engram::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::engram::event::execute_hash_rows{request, result}));
  CHECK(
      machine.is(stateforward::sml::state<emel::kernel::engram::state_ready>));
}
