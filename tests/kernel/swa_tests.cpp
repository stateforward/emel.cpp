#include <array>
#include <doctest/doctest.h>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/swa/sm.hpp"

namespace {

using emel::kernel::swa::event::dispatch_result;

} // namespace

TEST_CASE("swa attend computes grouped sliding-window softmax attention") {
  // heads=2, kv_heads=1, head_dim=2, capacity=4, positions 0..2 valid.
  std::array<float, 8> key_cache{};
  std::array<float, 8> value_cache{};
  const auto put = [](std::array<float, 8> &cache, const uint32_t position,
                      const float a, const float b) {
    cache[position * 2u] = a;
    cache[position * 2u + 1u] = b;
  };
  put(key_cache, 0u, 1.0f, 1.0f);
  put(key_cache, 1u, 0.0f, 2.0f);
  put(key_cache, 2u, -1.0f, 0.5f);
  put(value_cache, 0u, 1.0f, 0.0f);
  put(value_cache, 1u, 0.0f, 1.0f);
  put(value_cache, 2u, 2.0f, 2.0f);

  const std::array<float, 4> query{1.0f, 0.0f, 0.5f, -0.5f};
  std::array<float, 4> workspace{};
  std::array<float, 4> output{};
  const emel::kernel::swa::event::attend_request request{.query = query,
                                                         .key_cache = key_cache,
                                                         .value_cache =
                                                             value_cache,
                                                         .position = 2u,
                                                         .window_begin = 0u,
                                                         .capacity = 4u,
                                                         .heads = 2u,
                                                         .kv_heads = 1u,
                                                         .head_dim = 2u,
                                                         .workspace = workspace,
                                                         .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_attend{request, result}));
  CHECK(output[0] == doctest::Approx(0.8560338020324707f));
  CHECK(output[1] == doctest::Approx(0.5640538930892944f));
  CHECK(output[2] == doctest::Approx(1.0458049774169922f));
  CHECK(output[3] == doctest::Approx(0.8022611737251282f));
}

TEST_CASE("swa cache write lands rows at position modulo capacity") {
  std::array<float, 8> key_cache{};
  std::array<float, 8> value_cache{};
  const std::array<float, 2> key_rows{5.0f, 6.0f};
  const std::array<float, 2> value_rows{7.0f, 8.0f};
  const emel::kernel::swa::event::cache_write_request request{
      .key_rows = key_rows,
      .value_rows = value_rows,
      .position = 6u, // capacity 4 -> physical slot 2
      .capacity = 4u,
      .kv_heads = 1u,
      .head_dim = 2u,
      .key_cache = key_cache,
      .value_cache = value_cache};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_cache_write{request, result}));
  CHECK(key_cache[4] == 5.0f);
  CHECK(key_cache[5] == 6.0f);
  CHECK(value_cache[4] == 7.0f);
  CHECK(value_cache[5] == 8.0f);
}

TEST_CASE("swa gate mul applies elementwise sigmoid gating") {
  std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> gate_logits{0.3f, -1.2f, 2.0f, 0.0f};
  const emel::kernel::swa::event::gate_mul_request request{
      .values = values, .gate_logits = gate_logits, .dim = 4u};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_gate_mul{request, result}));
  CHECK(values[0] == doctest::Approx(0.5744425058364868f));
  CHECK(values[1] == doctest::Approx(0.4629504382610321f));
  CHECK(values[2] == doctest::Approx(2.6423912048339844f));
  CHECK(values[3] == doctest::Approx(2.0f));
}

TEST_CASE("swa residual gate adds sigmoid-scaled values onto skip") {
  // gate = fp16(0.7) decoded to f32.
  const float gate = emel::kernel::detail::quant::fp16_to_fp32(0x399au);
  const std::array<float, 2> skip{1.0f, -1.0f};
  const std::array<float, 2> values{2.0f, 4.0f};
  std::array<float, 2> output{};
  const emel::kernel::swa::event::residual_gate_request request{
      .skip = skip,
      .gate = gate,
      .values = values,
      .dim = 2u,
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::swa::event::execute_residual_gate{request, result}));
  CHECK(output[0] == doctest::Approx(2.3364620208740234f));
  CHECK(output[1] == doctest::Approx(1.672924280166626f));
}

TEST_CASE("swa attend guard rejects windows wider than the ring capacity") {
  const std::array<float, 4> query{};
  const std::array<float, 8> key_cache{};
  const std::array<float, 8> value_cache{};
  std::array<float, 8> workspace{};
  std::array<float, 4> output{};
  const emel::kernel::swa::event::attend_request request{
      .query = query,
      .key_cache = key_cache,
      .value_cache = value_cache,
      .position = 5u,
      .window_begin = 0u, // span 6 > capacity 4
      .capacity = 4u,
      .heads = 2u,
      .kv_heads = 1u,
      .head_dim = 2u,
      .workspace = workspace,
      .output = output};
  emel::kernel::swa::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::swa::event::execute_attend{request, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::swa::state_ready>));
}
