#include <array>
#include <cstdint>
#include <cstring>
#include <doctest/doctest.h>

#include "emel/kernel/mhc/sm.hpp"

namespace {

using emel::kernel::mhc::event::dispatch_result;

template <size_t count>
std::array<uint8_t, count * 2u>
pack_f16(const std::array<uint16_t, count> &bits) {
  std::array<uint8_t, count * 2u> bytes{};
  std::memcpy(bytes.data(), bits.data(), bytes.size());
  return bytes;
}

} // namespace

TEST_CASE("mhc pre mix weights lanes by sigmoid(a*dot + b + pre_off)") {
  // n=2, dim=2, layer=1 (lane 1): a=fp16(0.5), b=fp16([0.1,-0.2]);
  // lanes [[1,2],[3,-1]], phi_dots [0.4,-0.6].
  const std::array<float, 4> lanes{1.0f, 2.0f, 3.0f, -1.0f};
  const std::array<float, 2> phi_dots{0.4f, -0.6f};
  const auto a = pack_f16<1u>({0x3800u});
  const auto b = pack_f16<2u>({0x2e66u, 0xb266u});
  std::array<float, 2> output{};
  const emel::kernel::mhc::event::pre_mix_request request{.lanes = lanes,
                                                          .phi_dots = phi_dots,
                                                          .a = a,
                                                          .b = b,
                                                          .lane_index = 1u,
                                                          .lane_count = 2u,
                                                          .dim = 2u,
                                                          .output = output};
  emel::kernel::mhc::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::mhc::event::execute_pre_mix{request, result}));
  CHECK(output[0] == doctest::Approx(2.9361939430236816f));
  CHECK(output[1] == doctest::Approx(-0.9224362373352051f));
}

TEST_CASE("mhc post mix routes lanes through 20-iteration sinkhorn") {
  // n=2, dim=2, layer=1 (lane 1); expectations from the reference `_sinkhorn`
  // and hpost formulas with fp16-decoded a/b payloads.
  const std::array<float, 4> lanes{1.0f, 2.0f, 3.0f, -1.0f};
  const std::array<float, 2> block_out{2.0f, 1.0f};
  const std::array<float, 2> u{0.5f, 0.5f}; // y = block_out - u = [1.5, 0.5]
  const std::array<float, 2> post_dots{0.2f, -0.3f};
  const std::array<float, 4> res_dots{0.1f, 0.4f, -0.2f, 0.3f};
  const auto a_post = pack_f16<1u>({0x399au});          // 0.7
  const auto b_post = pack_f16<2u>({0x2a66u, 0xae66u}); // [0.05, -0.1]
  const auto a_res = pack_f16<1u>({0x3e00u});           // 1.5
  const auto b_res = pack_f16<4u>({0x3266u, 0xae66u, 0x0000u, 0x34cdu});
  std::array<float, 4> output{};
  const emel::kernel::mhc::event::post_mix_request request{
      .lanes = lanes,
      .block_out = block_out,
      .u = u,
      .post_dots = post_dots,
      .res_dots = res_dots,
      .a_post = a_post,
      .b_post = b_post,
      .a_res = a_res,
      .b_res = b_res,
      .lane_index = 1u,
      .lane_count = 2u,
      .dim = 2u,
      .output = output};
  emel::kernel::mhc::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::mhc::event::execute_post_mix{request, result}));
  CHECK(output[0] == doctest::Approx(1.8437339067459106f));
  CHECK(output[1] == doctest::Approx(0.8535779118537903f));
  CHECK(output[2] == doctest::Approx(3.4905920028686523f));
  CHECK(output[3] == doctest::Approx(0.5911972522735596f));
}

TEST_CASE("mhc mean lanes averages lane rows") {
  const std::array<float, 4> lanes{1.0f, 2.0f, 3.0f, -1.0f};
  std::array<float, 2> output{};
  const emel::kernel::mhc::event::mean_lanes_request request{
      .lanes = lanes, .lane_count = 2u, .dim = 2u, .output = output};
  emel::kernel::mhc::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::mhc::event::execute_mean_lanes{request, result}));
  CHECK(output[0] == doctest::Approx(2.0f));
  CHECK(output[1] == doctest::Approx(0.5f));
}

TEST_CASE("mhc pre mix guard rejects lane index out of range") {
  const std::array<float, 4> lanes{};
  const std::array<float, 2> phi_dots{};
  const std::array<uint8_t, 2> a{};
  const std::array<uint8_t, 4> b{};
  std::array<float, 2> output{};
  const emel::kernel::mhc::event::pre_mix_request request{.lanes = lanes,
                                                          .phi_dots = phi_dots,
                                                          .a = a,
                                                          .b = b,
                                                          .lane_index = 2u,
                                                          .lane_count = 2u,
                                                          .dim = 2u,
                                                          .output = output};
  emel::kernel::mhc::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::mhc::event::execute_pre_mix{request, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::mhc::state_ready>));
}
