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

template <uint32_t Bits>
constexpr size_t row_bytes(const uint32_t n) {
  return emel::kernel::cq::detail::packed_row_bytes<Bits>(n);
}
std::vector<uint8_t> make_blob(const std::vector<uint32_t> &indices,
                               uint32_t out, uint32_t in, uint32_t group,
                               uint32_t bits, const std::vector<uint16_t> &norms) {
  const uint32_t pad = (in + group - 1u) / group * group;
  const size_t rb = bits == 2u ? row_bytes<2u>(pad) : bits == 3u ? row_bytes<3u>(pad) : bits == 4u ? row_bytes<4u>(pad) : row_bytes<5u>(pad);
  const size_t nc = static_cast<size_t>(out) * pad / group;
  std::vector<uint8_t> blob(static_cast<size_t>(out) * rb + nc * 2u);
  for (uint32_t r = 0u; r < out; ++r) for (uint32_t i = 0u; i < pad; ++i) {
    const uint32_t index = indices[static_cast<size_t>(r) * pad + i];
    uint8_t *p = blob.data() + static_cast<size_t>(r) * rb;
    if (bits == 5u) {
      const uint32_t crumb = index == 0u ? 3u : index - 1u;
      p[i >> 2u] |= static_cast<uint8_t>(crumb << ((i & 3u) * 2u));
    } else {
      const size_t bit = static_cast<size_t>(i) * bits;
      p[bit >> 3u] |= static_cast<uint8_t>(index << (bit & 7u));
      if ((bit & 7u) + bits > 8u) p[(bit >> 3u) + 1u] |= static_cast<uint8_t>(index >> (8u - (bit & 7u)));
    }
  }
  const size_t po = static_cast<size_t>(out) * rb;
  for (size_t i = 0u; i < nc; ++i) { blob[po + i * 2u] = static_cast<uint8_t>(norms[i]); blob[po + i * 2u + 1u] = static_cast<uint8_t>(norms[i] >> 8u); }
  return blob;
}
tensor_view view_for(const std::vector<uint8_t> &blob, uint32_t out, uint32_t in, uint32_t group, uint32_t bits) {
  return tensor_view{.dtype = 3u, .ndim = 2u, .shape = {out, in, 0u, 0u}, .nbytes = blob.size(), .group = group, .bits = bits, .data = blob.data()};
}
}

TEST_CASE("CQ2 scalar parity applies LSB-first indices and normalized FWHT") {
  std::array<float, 28u> cb{}; cb[0] = -.5f; cb[1] = -.1f; cb[2] = .1f; cb[3] = .5f;
  const std::vector<uint32_t> ix{0u,1u,2u,3u,0u,1u,2u,3u}; const auto blob = make_blob(ix,1u,8u,8u,2u,{0x3c00}); const auto view = view_for(blob,1u,8u,8u,2u);
  const std::array<float,8u> a{1,2,3,4,5,6,7,8}; std::array<float,8u> w{}; std::array<float,1u> o{}; gemv_request q{view,cb,a,o,w}; emel::kernel::cq::sm sm; emel::kernel::cq::event::dispatch_result r{};
  REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q2{q,r})); std::array<float,8u> t=a; emel::kernel::cq::detail::fwht(t.data(),8u); float e=0; for(uint32_t i=0;i<8u;++i)e+=cb[ix[i]]*t[i]; CHECK(o[0]==doctest::Approx(e));
}
TEST_CASE("CQ ternary scalar parity uses crumb encoding") {
  const std::vector<uint32_t> ix{0u,1u,2u,0u,2u,1u,0u,2u}; const auto blob=make_blob(ix,1u,8u,8u,5u,{0x3c00}); const auto view=view_for(blob,1u,8u,8u,5u); const std::array<float,8u> a{1,0,0,0,0,0,0,0}; std::array<float,8u>w{}; std::array<float,1u>o{}; std::array<float,28u>cb{}; gemv_request q{view,cb,a,o,w}; emel::kernel::cq::sm sm; emel::kernel::cq::event::dispatch_result r{}; REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_ternary{q,r})); std::array<float,8u>t=a; emel::kernel::cq::detail::fwht(t.data(),8u); float e=0; for(uint32_t i=0;i<8u;++i)e+=emel::kernel::cq::detail::code_value<5u>(ix[i],8u,cb)*t[i]; CHECK(o[0]==doctest::Approx(e));
}
TEST_CASE("CQ3 and CQ4 explicit scalar and AVX2 routes preserve parity") {
  for (uint32_t bits : {3u,4u}) { constexpr uint32_t in=16u, group=8u; std::array<float,28u>cb{}; const uint32_t off=bits==3u?4u:12u, levels=1u<<bits; for(uint32_t i=0;i<levels;++i)cb[off+i]=static_cast<float>(i+1u)/10.f; std::vector<uint32_t>ix; for(uint32_t i=0;i<in*2u;++i)ix.push_back(i%levels); const auto blob=make_blob(ix,2u,in,group,bits,{0x3c00,0x4000,0x3c00,0x4000}); const auto view=view_for(blob,2u,in,group,bits); std::array<float,in>a{};for(uint32_t i=0;i<in;++i)a[i]=static_cast<float>(i+1u)/8.f;std::array<float,in>w{};std::array<float,2u>so{};gemv_request q{view,cb,a,so,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result r{};if(bits==3u)REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q3{q,r}));else REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{q,r}));
#if defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { std::array<float,2u>ao{};gemv_request aq{view,cb,a,ao,w};emel::kernel::cq::sm am;emel::kernel::cq::event::dispatch_result ar{};if(bits==3u)REQUIRE(am.process_event(emel::kernel::cq::event::execute_avx2_q3{aq,ar}));else REQUIRE(am.process_event(emel::kernel::cq::event::execute_avx2_q4{aq,ar}));CHECK(ao[0]==doctest::Approx(so[0]));CHECK(ao[1]==doctest::Approx(so[1])); }
#endif
  }
}
TEST_CASE("CQ route guard rejects undersized padded workspace") {
 const std::vector<uint32_t> ix(16u,0u);const auto blob=make_blob(ix,1u,13u,8u,4u,{0x3c00,0x3c00});const auto view=view_for(blob,1u,13u,8u,4u);std::array<float,28u>cb{};std::array<float,13u>a{};std::array<float,1u>o{};std::array<float,13u>w{};gemv_request q{view,cb,a,o,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result r{};CHECK_FALSE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{q,r}));CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));
}
