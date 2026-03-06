#pragma once

#include <cstdint>
#include <type_traits>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::test {

using dtype = emel::kernel::event::dtype;
using tensor_view = emel::kernel::event::tensor_view;
using tensor_view_mut = emel::kernel::event::tensor_view_mut;

template <class tensor_type>
inline void fill_default_nb(tensor_type & tensor) {
  const auto elem_size =
      static_cast<uint64_t>(emel::kernel::detail::dtype_size_bytes(
          emel::kernel::detail::dtype_code(tensor.type)));
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

inline tensor_view make_src(const void * data, const dtype type, const uint64_t ne0,
                            const uint64_t ne1 = 1, const uint64_t ne2 = 1,
                            const uint64_t ne3 = 1) {
  tensor_view out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(out);
  return out;
}

inline tensor_view_mut make_dst(void * data, const dtype type, const uint64_t ne0,
                                const uint64_t ne1 = 1, const uint64_t ne2 = 1,
                                const uint64_t ne3 = 1) {
  tensor_view_mut out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(out);
  return out;
}

template <class event_type>
inline event_type make_smoke_op_event() {
  event_type ev{};
  static float src0[16] = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
      12.0f, 13.0f, 14.0f, 15.0f,
  };
  static float src1[16] = {
      15.0f, 14.0f, 13.0f, 12.0f,
      11.0f, 10.0f, 9.0f, 8.0f,
      7.0f, 6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f, 0.0f,
  };
  static float src2[16] = {
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
  };
  static float dst[16] = {};

  ev.src0 = make_src(src0, dtype::f32, 4);
  ev.src1 = make_src(src1, dtype::f32, 4);
  ev.src2 = make_src(src2, dtype::f32, 4);
  ev.dst = make_dst(dst, dtype::f32, 4);
  ev.ith = 0;
  ev.nth = 1;

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_mul_mat>) {
    ev.src0 = make_src(src0, dtype::f32, 2, 2);
    ev.src1 = make_src(src1, dtype::f32, 2, 2);
    ev.dst = make_dst(dst, dtype::f32, 2, 2);
  }

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_soft_max>) {
    ev.src0 = make_src(src0, dtype::f32, 4, 2);
    ev.dst = make_dst(dst, dtype::f32, 4, 2);
  }

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_unary>) {
    ev.subop = emel::kernel::event::unary_subop::abs;
  }

  return ev;
}

}  // namespace emel::kernel::test
