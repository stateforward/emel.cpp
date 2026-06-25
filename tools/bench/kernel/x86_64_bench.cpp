#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/x86_64/sm.hpp"

#include "kernel/bench_common.hpp"

namespace {

using dtype = emel::kernel::event::dtype;
using emel::kernel::detail::quant::QK_K;
using emel::kernel::detail::quant::block_q2_k;
using emel::kernel::detail::quant::block_q3_k;
using emel::kernel::detail::quant::block_q6_k;

constexpr size_t k_x86_quantized_block_count = 2u;
constexpr uint64_t k_x86_quantized_k = QK_K * k_x86_quantized_block_count;
constexpr uint64_t k_x86_quantized_rows = 2u;
constexpr uint64_t k_x86_quantized_cols = 2u;

template <class tensor_type>
void fill_x86_dense_nb(tensor_type & tensor, const uint64_t elem_size) {
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_x86_quantized_src(
    const void * data,
    const dtype type,
    const uint64_t ne0,
    const uint64_t ne1) {
  emel::kernel::event::tensor_view out{};
  const size_t row_bytes = emel::kernel::detail::quantized_row_storage_bytes(
      emel::kernel::detail::dtype_code(type), ne0);
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, 1u, 1u};
  out.nb[0] = 1u;
  out.nb[1] = row_bytes;
  out.nb[2] = row_bytes * ne1;
  out.nb[3] = out.nb[2];
  return out;
}

emel::kernel::event::tensor_view make_x86_src_view(
    const void * data,
    const dtype type,
    const uint64_t ne0,
    const uint64_t ne1 = 1u,
    const uint64_t ne2 = 1u) {
  emel::kernel::event::tensor_view out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, 1u};
  fill_x86_dense_nb(out, emel::kernel::detail::dtype_size_bytes(
                             emel::kernel::detail::dtype_code(type)));
  return out;
}

emel::kernel::event::tensor_view_mut make_x86_dst_view(
    void * data,
    const dtype type,
    const uint64_t ne0,
    const uint64_t ne1 = 1u,
    const uint64_t ne2 = 1u) {
  emel::kernel::event::tensor_view_mut out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, 1u};
  fill_x86_dense_nb(out, emel::kernel::detail::dtype_size_bytes(
                             emel::kernel::detail::dtype_code(type)));
  return out;
}

void fill_x86_q2_block(block_q2_k & q2, const uint32_t salt) {
  q2.d = static_cast<uint16_t>(0x3c00u + (salt % 17u));
  q2.dmin = static_cast<uint16_t>(0x3800u + (salt % 11u));
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>((((i + salt) % 13u) << 4u) |
                                        (((i * 5u) + salt) % 15u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * (23u + salt)) ^ ((i + salt) >> 1u));
  }
}

void fill_x86_q3_block(block_q3_k & q3, const uint32_t salt) {
  q3.d = static_cast<uint16_t>(0x3c00u + (salt % 19u));
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * (17u + salt)) ^ (0x5au + salt));
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * (9u + salt)) ^ (0xa5u - salt));
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * (13u + salt)) ^ (0x33u + salt * 7u));
  }
}

void fill_x86_q6_block(block_q6_k & q6, const uint32_t salt) {
  q6.d = static_cast<uint16_t>(0x3c00u + (salt % 23u));
  for (size_t i = 0; i < q6.scales.size(); ++i) {
    const int32_t scale_value =
        static_cast<int32_t>(((i + salt) * 3u) % 31u) - 15;
    q6.scales[i] = static_cast<int8_t>(scale_value);
  }
  for (size_t i = 0; i < q6.ql.size(); ++i) {
    q6.ql[i] = static_cast<uint8_t>((i * (19u + salt)) ^ (0x6cu + salt));
  }
  for (size_t i = 0; i < q6.qh.size(); ++i) {
    q6.qh[i] = static_cast<uint8_t>((i * (7u + salt)) ^ (0x95u - salt));
  }
}

std::array<float, k_x86_quantized_k * k_x86_quantized_cols>
make_x86_quantized_rhs_values(const uint32_t salt) {
  std::array<float, k_x86_quantized_k * k_x86_quantized_cols> rhs{};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * (5u + salt)) % 43u) - 21;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }
  return rhs;
}

template <class block_type, class fill_block_fn>
std::array<block_type, k_x86_quantized_rows * k_x86_quantized_block_count>
make_x86_quantized_rows(fill_block_fn fill_block, const uint32_t salt) {
  std::array<block_type, k_x86_quantized_rows * k_x86_quantized_block_count>
      rows{};
  for (size_t idx = 0; idx < rows.size(); ++idx) {
    fill_block(rows[idx], static_cast<uint32_t>(idx) + salt);
  }
  return rows;
}

template <class block_type, class optimized_counter_fn,
          class shared_counter_fn>
void append_emel_x86_quantized_case(std::vector<emel::bench::result> & results,
                                    const emel::bench::config & cfg,
                                    const char * case_name,
                                    const dtype block_type_code,
                                    const block_type * blocks,
                                    const float * rhs,
                                    optimized_counter_fn optimized_counter,
                                    shared_counter_fn shared_counter) {
  std::array<float, k_x86_quantized_rows * k_x86_quantized_cols> dst{};
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_x86_quantized_src(blocks,
                                     block_type_code,
                                     k_x86_quantized_k,
                                     k_x86_quantized_rows),
      .src1 = make_x86_src_view(rhs, dtype::f32, k_x86_quantized_cols,
                                k_x86_quantized_k),
      .dst = make_x86_dst_view(dst.data(), dtype::f32, k_x86_quantized_cols,
                               k_x86_quantized_rows),
      .nth = 1,
  };
  emel::kernel::x86_64::sm machine{};
  volatile float sink = 0.0f;

  auto fn = [&]() {
    const uint64_t optimized_before = optimized_counter(machine);
    const uint64_t shared_before = shared_counter(machine);
    const bool ok = machine.process_event(ev);
    if (!ok || optimized_counter(machine) != optimized_before + 1u ||
        shared_counter(machine) != shared_before) {
      std::abort();
    }
    sink += dst[0];
  };
  results.push_back(emel::bench::measure_case(case_name, cfg, fn));
  (void)sink;
}

template <class block_type>
void append_reference_x86_quantized_case(
    std::vector<emel::bench::result> & results,
    const emel::bench::config & cfg,
    const char * case_name,
    const dtype block_type_code,
    const block_type * blocks,
    const float * rhs) {
  std::array<float, k_x86_quantized_rows * k_x86_quantized_cols> dst{};
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_x86_quantized_src(blocks,
                                     block_type_code,
                                     k_x86_quantized_k,
                                     k_x86_quantized_rows),
      .src1 = make_x86_src_view(rhs, dtype::f32, k_x86_quantized_cols,
                                k_x86_quantized_k),
      .dst = make_x86_dst_view(dst.data(), dtype::f32, k_x86_quantized_cols,
                               k_x86_quantized_rows),
      .nth = 1,
  };
  volatile float sink = 0.0f;

  auto fn = [&]() {
    const bool ok = emel::kernel::detail::execute_scalar(ev);
    if (!ok) {
      std::abort();
    }
    sink += dst[0];
  };
  results.push_back(emel::bench::measure_case(case_name, cfg, fn));
  (void)sink;
}

struct x86_flash_fixture {
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  uint16_t k[8] = {
      0x3c00u, 0x0000u, 0x0000u, 0x0000u,
      0x0000u, 0x3c00u, 0x0000u, 0x0000u,
  };
  uint16_t v[8] = {
      0x4000u, 0x0000u, 0x0000u, 0x0000u,
      0x0000u, 0x4400u, 0x0000u, 0x0000u,
  };
  float dst[4] = {};
};

emel::kernel::event::op_flash_attn_ext make_x86_flash_event(
    x86_flash_fixture & fixture) {
  emel::kernel::event::op_flash_attn_ext ev{};
  ev.src0 = make_x86_src_view(fixture.q, dtype::f32, 4u, 1u, 1u);
  ev.src1 = make_x86_src_view(fixture.k, dtype::f16, 4u, 2u, 1u);
  ev.src2 = make_x86_src_view(fixture.v, dtype::f16, 4u, 2u, 1u);
  ev.dst = make_x86_dst_view(fixture.dst, dtype::f32, 4u, 1u, 1u);
  ev.nth = 1;
  const float scale = 1.0f;
  std::memcpy(ev.op_params.data(), &scale, sizeof(scale));
  ev.op_params_size = sizeof(scale);
  return ev;
}

void append_emel_x86_flash_case(std::vector<emel::bench::result> & results,
                                const emel::bench::config & cfg) {
  x86_flash_fixture fixture{};
  const emel::kernel::event::op_flash_attn_ext ev =
      make_x86_flash_event(fixture);
  emel::kernel::x86_64::sm machine{};
  volatile float sink = 0.0f;

  auto fn = [&]() {
    const uint64_t optimized_before = machine.optimized_flash_dispatch_count();
    const uint64_t shared_before = machine.shared_flash_dispatch_count();
    const bool ok = machine.process_event(ev);
    if (!ok || machine.optimized_flash_dispatch_count() !=
                   optimized_before + 1u ||
        machine.shared_flash_dispatch_count() != shared_before) {
      std::abort();
    }
    sink += fixture.dst[0];
  };
  results.push_back(emel::bench::measure_case(
      "kernel/x86_64/op_flash_attn_ext_decode_like", cfg, fn));
  (void)sink;
}

void append_reference_x86_flash_case(std::vector<emel::bench::result> & results,
                                     const emel::bench::config & cfg) {
  x86_flash_fixture fixture{};
  const emel::kernel::event::op_flash_attn_ext ev =
      make_x86_flash_event(fixture);
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = false,
      .fma_available = false,
      .f16c_available = false,
  };
  emel::kernel::x86_64::sm machine{
      emel::kernel::x86_64::action::context{contract, {}, 0}};
  volatile float sink = 0.0f;

  auto fn = [&]() {
    const uint64_t shared_before = machine.shared_flash_dispatch_count();
    const bool ok = machine.process_event(ev);
    if (!ok || machine.optimized_flash_dispatch_count() != 0u ||
        machine.shared_flash_dispatch_count() != shared_before + 1u) {
      std::abort();
    }
    sink += fixture.dst[0];
  };
  results.push_back(emel::bench::measure_case(
      "kernel/x86_64/op_flash_attn_ext_decode_like", cfg, fn));
  (void)sink;
}

void append_emel_x86_optimized_cases(std::vector<emel::bench::result> & results,
                                     const emel::bench::config & cfg) {
  const auto q2_rows =
      make_x86_quantized_rows<block_q2_k>(fill_x86_q2_block, 11u);
  const auto q3_rows =
      make_x86_quantized_rows<block_q3_k>(fill_x86_q3_block, 19u);
  const auto q6_rows =
      make_x86_quantized_rows<block_q6_k>(fill_x86_q6_block, 37u);
  const auto rhs = make_x86_quantized_rhs_values(3u);

  append_emel_x86_flash_case(results, cfg);
  append_emel_x86_quantized_case(
      results, cfg, "kernel/x86_64/op_mul_mat_q2_k_q8_k", dtype::q2_k,
      q2_rows.data(), rhs.data(),
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.optimized_q2_dispatch_count();
      },
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.shared_q2_dispatch_count();
      });
  append_emel_x86_quantized_case(
      results, cfg, "kernel/x86_64/op_mul_mat_q3_k_q8_k", dtype::q3_k,
      q3_rows.data(), rhs.data(),
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.optimized_q3_dispatch_count();
      },
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.shared_q3_dispatch_count();
      });
  append_emel_x86_quantized_case(
      results, cfg, "kernel/x86_64/op_mul_mat_q6_k_q8_k", dtype::q6_k,
      q6_rows.data(), rhs.data(),
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.optimized_q6_dispatch_count();
      },
      [](const emel::kernel::x86_64::sm & machine) {
        return machine.shared_q6_dispatch_count();
      });
}

void append_reference_x86_optimized_cases(
    std::vector<emel::bench::result> & results,
    const emel::bench::config & cfg) {
  const auto q2_rows =
      make_x86_quantized_rows<block_q2_k>(fill_x86_q2_block, 11u);
  const auto q3_rows =
      make_x86_quantized_rows<block_q3_k>(fill_x86_q3_block, 19u);
  const auto q6_rows =
      make_x86_quantized_rows<block_q6_k>(fill_x86_q6_block, 37u);
  const auto rhs = make_x86_quantized_rhs_values(3u);

  append_reference_x86_flash_case(results, cfg);
  append_reference_x86_quantized_case(results, cfg,
                                      "kernel/x86_64/op_mul_mat_q2_k_q8_k",
                                      dtype::q2_k, q2_rows.data(), rhs.data());
  append_reference_x86_quantized_case(results, cfg,
                                      "kernel/x86_64/op_mul_mat_q3_k_q8_k",
                                      dtype::q3_k, q3_rows.data(), rhs.data());
  append_reference_x86_quantized_case(results, cfg,
                                      "kernel/x86_64/op_mul_mat_q6_k_q8_k",
                                      dtype::q6_k, q6_rows.data(), rhs.data());
}

}  // namespace

namespace emel::bench {

void append_emel_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  emel::kernel::x86_64::sm x86_machine{};
  auto exec = [&](const auto & ev) {
    return x86_machine.process_event(ev);
  };
  append_emel_backend_cases(results, cfg, "x86_64", exec);
  append_emel_x86_optimized_cases(results, cfg);
}

void append_reference_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "x86_64");
  append_reference_x86_optimized_cases(results, cfg);
}

}  // namespace emel::bench
