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
template <uint32_t Bits> std::vector<uint8_t> blob(const std::vector<uint32_t>& ix,uint32_t out,uint32_t in,uint32_t group,const std::vector<uint16_t>& ns){const uint32_t pad=(in+group-1u)/group*group;const size_t rb=emel::kernel::cq::detail::packed_row_bytes<Bits>(pad);const size_t nc=static_cast<size_t>(out)*pad/group;std::vector<uint8_t>b(static_cast<size_t>(out)*rb+nc*2u);for(uint32_t r=0;r<out;++r)for(uint32_t i=0;i<pad;++i){uint8_t*p=b.data()+static_cast<size_t>(r)*rb;const uint32_t v=ix[static_cast<size_t>(r)*pad+i];if constexpr(Bits==5u){const uint32_t c=v==0u?3u:v-1u;p[i>>2u]|=static_cast<uint8_t>(c<<((i&3u)*2u));}else{const size_t bit=static_cast<size_t>(i)*Bits;p[bit>>3u]|=static_cast<uint8_t>(v<<(bit&7u));if((bit&7u)+Bits>8u)p[(bit>>3u)+1u]|=static_cast<uint8_t>(v>>(8u-(bit&7u)));}}const size_t po=static_cast<size_t>(out)*rb;for(size_t i=0;i<nc;++i){b[po+i*2u]=static_cast<uint8_t>(ns[i]);b[po+i*2u+1u]=static_cast<uint8_t>(ns[i]>>8u);}return b;}
tensor_view view(const std::vector<uint8_t>&b,uint32_t out,uint32_t in,uint32_t group,uint32_t bits){return tensor_view{.dtype=3u,.ndim=2u,.shape={out,in,0u,0u},.nbytes=b.size(),.group=group,.bits=bits,.data=b.data()};}
template<uint32_t Bits> void run_route(uint32_t in,const std::array<float,28u>&cb){constexpr uint32_t group=8u;const uint32_t levels=1u<<Bits;std::vector<uint32_t>ix;for(uint32_t i=0;i<4u*in;++i)ix.push_back(i%levels);const auto b=blob<Bits>(ix,4u,in,group,{0x3c00,0x4000,0x3c00,0x4000,0x3c00,0x4000,0x3c00,0x4000});const auto v=view(b,4u,in,group,Bits);std::array<float,32u>a{};for(uint32_t i=0;i<in;++i)a[i]=static_cast<float>(i+1u)/8.f;std::array<float,32u>w{};std::array<float,4u>so{};gemv_request q{v,cb,a,so,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result sr{};REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar<Bits>{q,sr}));CHECK(sr.accepted);
#if defined(__AVX2__) && defined(__FMA__)
std::array<float,4u>ao{};gemv_request aq{v,cb,a,ao,w};emel::kernel::cq::sm am;emel::kernel::cq::event::dispatch_result ar{};REQUIRE(am.process_event(emel::kernel::cq::event::execute_avx2<Bits>{aq,ar}));for(uint32_t r=0;r<4u;++r)CHECK(ao[r]==doctest::Approx(so[r]));
#endif
}
}
TEST_CASE("CQ2 scalar parity and normalized FWHT"){std::array<float,28u>cb{};cb[0]=-.5f;cb[1]=-.1f;cb[2]=.1f;cb[3]=.5f;const std::vector<uint32_t>ix{0,1,2,3,0,1,2,3};const auto b=blob<2u>(ix,1,8,8,{0x3c00});const auto v=view(b,1,8,8,2);const std::array<float,8u>a{1,2,3,4,5,6,7,8};std::array<float,8u>w{};std::array<float,1u>o{};gemv_request q{v,cb,a,o,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result r{};REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_q2{q,r}));std::array<float,8u>t=a;emel::kernel::cq::detail::fwht(t.data(),8);float e=0;for(uint32_t i=0;i<8;++i)e+=cb[ix[i]]*t[i];CHECK(o[0]==doctest::Approx(e));}
TEST_CASE("CQ3 and CQ4 explicit routes preserve parity"){std::array<float,28u>cb{};for(uint32_t i=0;i<8;++i)cb[4+i]=static_cast<float>(i+1)/10.f;run_route<3u>(16u,cb);for(uint32_t i=0;i<16;++i)cb[12+i]=static_cast<float>(i+1)/10.f;run_route<4u>(16u,cb);}
TEST_CASE("CQ ternary crumbs use analytic centroid"){const std::vector<uint32_t>ix{0,1,2,0,2,1,0,2};const auto b=blob<5u>(ix,1,8,8,{0x3c00});const auto v=view(b,1,8,8,5);const std::array<float,8u>a{1,0,0,0,0,0,0,0};std::array<float,8u>w{};std::array<float,1u>o{};std::array<float,28u>cb{};gemv_request q{v,cb,a,o,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result r{};REQUIRE(sm.process_event(emel::kernel::cq::event::execute_scalar_ternary{q,r}));}
TEST_CASE("CQ guard rejects incomplete padded workspace"){const auto b=blob<4u>(std::vector<uint32_t>(16,0),1,13,8,{0x3c00,0x3c00});const auto v=view(b,1,13,8,4);std::array<float,28u>cb{};std::array<float,13u>a{};std::array<float,1u>o{};std::array<float,13u>w{};gemv_request q{v,cb,a,o,w};emel::kernel::cq::sm sm;emel::kernel::cq::event::dispatch_result r{};CHECK_FALSE(sm.process_event(emel::kernel::cq::event::execute_scalar_q4{q,r}));CHECK(sm.is(stateforward::sml::state<emel::kernel::cq::state_ready>));}
