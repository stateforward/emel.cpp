#include "bench_cases.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/detail.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/detail.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

using emel::kernel::event::dtype;

constexpr int64_t k_vec_len = 1024;
constexpr int64_t k_softmax_width = 128;
constexpr int64_t k_softmax_rows = 8;
constexpr int64_t k_mm_k = 64;
constexpr int64_t k_mm_m = 32;
constexpr int64_t k_mm_n = 48;

std::string make_case_name(const char * backend, const char * op_name) {
  return std::string{"kernel/"} + backend + "/" + op_name;
}

std::vector<float> make_signed_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const float wave = std::sin(static_cast<float>(i) * 0.013f) * scale;
    const float bucket = static_cast<float>((i % 29) - 14) * 0.03125f;
    out[static_cast<size_t>(i)] = wave + bucket + bias;
  }
  return out;
}

std::vector<float> make_positive_data(const int64_t count, const float scale, const float bias) {
  std::vector<float> out = make_signed_data(count, scale, bias);
  for (float & value : out) {
    value = std::fabs(value) + 0.5f;
  }
  return out;
}

template <class tensor_type>
void fill_default_nb(tensor_type & tensor) {
  constexpr uint64_t elem_size = sizeof(float);
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_src_view(const float * data,
                                               const uint64_t ne0,
                                               const uint64_t ne1 = 1,
                                               const uint64_t ne2 = 1,
                                               const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

emel::kernel::event::tensor_view_mut make_dst_view(float * data,
                                                   const uint64_t ne0,
                                                   const uint64_t ne1 = 1,
                                                   const uint64_t ne2 = 1,
                                                   const uint64_t ne3 = 1) {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = dtype::f32;
  tensor.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(tensor);
  return tensor;
}

struct ggml_graph_case {
  std::vector<uint8_t> arena;
  ggml_context * ctx = nullptr;
  ggml_tensor * out = nullptr;
  ggml_cgraph * graph = nullptr;

  explicit ggml_graph_case(const size_t arena_bytes = 32u * 1024u * 1024u)
      : arena(arena_bytes) {
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = false;
    ctx = ggml_init(params);
  }

  ~ggml_graph_case() {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }

  bool compute() const {
    if (ctx == nullptr || graph == nullptr) {
      return false;
    }
    return ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
  }
};

void set_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values) {
  std::memcpy(ggml_get_data_f32(tensor), values.data(), values.size() * sizeof(float));
}

bool finalize_graph(ggml_graph_case & c, ggml_tensor * out) {
  c.out = out;
  c.graph = ggml_new_graph(c.ctx);
  if (c.graph == nullptr || c.out == nullptr) {
    return false;
  }
  ggml_build_forward_expand(c.graph, c.out);
  return c.compute();
}

template <class exec_fn>
void append_emel_backend_cases(std::vector<emel::bench::result> & results,
                               const emel::bench::config & cfg,
                               const char * backend,
                               exec_fn exec) {
  volatile float sink = 0.0f;

  {
    auto src = make_signed_data(k_vec_len, 1.25f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_dup ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[0] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_dup").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_add ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[1] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_add").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sub ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[2] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sub").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.25f, 0.75f);
    auto rhs = make_signed_data(k_vec_len, 0.45f, 0.5f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_mul ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[3] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_mul").c_str(), cfg, fn));
  }

  {
    auto lhs = make_positive_data(k_vec_len, 0.3f, 0.25f);
    auto rhs = make_positive_data(k_vec_len, 0.2f, 0.75f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_div ev{
      .src0 = make_src_view(lhs.data(), static_cast<uint64_t>(k_vec_len)),
      .src1 = make_src_view(rhs.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[4] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_div").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.5f, 0.125f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqr ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[5] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sqr").c_str(), cfg, fn));
  }

  {
    auto src = make_positive_data(k_vec_len, 0.35f, 0.2f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sqrt ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[6] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sqrt").c_str(), cfg, fn));
  }

  {
    auto src = make_positive_data(k_vec_len, 0.4f, 0.125f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_log ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[7] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_log").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_sin ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[8] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sin").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, -0.2f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_cos ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[9] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_cos").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_softmax_width * k_softmax_rows, 0.1f, 0.05f);
    std::vector<float> dst(static_cast<size_t>(k_softmax_width * k_softmax_rows));
    emel::kernel::event::op_soft_max ev{
      .src0 = make_src_view(src.data(),
                            static_cast<uint64_t>(k_softmax_width),
                            static_cast<uint64_t>(k_softmax_rows)),
      .dst = make_dst_view(dst.data(),
                           static_cast<uint64_t>(k_softmax_width),
                           static_cast<uint64_t>(k_softmax_rows)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[static_cast<size_t>(k_softmax_width)] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_soft_max").c_str(), cfg, fn));
  }

  {
    auto src0 = make_signed_data(k_mm_k * k_mm_m, 0.12f, 0.25f); // [m, k]
    auto matrix_a = make_signed_data(k_mm_k * k_mm_n, 0.08f, -0.1f); // [n, k]
    std::vector<float> src1(static_cast<size_t>(k_mm_k * k_mm_n));
    for (int64_t p = 0; p < k_mm_k; ++p) {
      for (int64_t j = 0; j < k_mm_n; ++j) {
        src1[static_cast<size_t>(p * k_mm_n + j)] = matrix_a[static_cast<size_t>(j * k_mm_k + p)];
      }
    }
    std::vector<float> dst(static_cast<size_t>(k_mm_n * k_mm_m));

    emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(src0.data(),
                            static_cast<uint64_t>(k_mm_k),
                            static_cast<uint64_t>(k_mm_m)),
      .src1 = make_src_view(src1.data(),
                            static_cast<uint64_t>(k_mm_n),
                            static_cast<uint64_t>(k_mm_k)),
      .dst = make_dst_view(dst.data(),
                           static_cast<uint64_t>(k_mm_n),
                           static_cast<uint64_t>(k_mm_m)),
      .nth = 1,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[0] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_mul_mat").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::neg,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[10] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_neg").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[11] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_relu").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.35f, 0.1f);
    std::vector<float> dst(static_cast<size_t>(k_vec_len));
    emel::kernel::event::op_unary ev{
      .src0 = make_src_view(src.data(), static_cast<uint64_t>(k_vec_len)),
      .dst = make_dst_view(dst.data(), static_cast<uint64_t>(k_vec_len)),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::exp,
    };

    auto fn = [&]() {
      const bool ok = exec(ev);
      sink += ok ? dst[12] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_exp").c_str(), cfg, fn));
  }

  (void) sink;
}

void append_reference_backend_cases(std::vector<emel::bench::result> & results,
                                    const emel::bench::config & cfg,
                                    const char * backend) {
  volatile float sink = 0.0f;

  {
    auto src = make_signed_data(k_vec_len, 1.25f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_dup(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[0] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_dup").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    const bool setup_ok = finalize_graph(c, ggml_add(c.ctx, a, b));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[1] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_add").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.75f, 0.5f);
    auto rhs = make_signed_data(k_vec_len, 0.55f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    const bool setup_ok = finalize_graph(c, ggml_sub(c.ctx, a, b));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[2] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sub").c_str(), cfg, fn));
  }

  {
    auto lhs = make_signed_data(k_vec_len, 0.25f, 0.75f);
    auto rhs = make_signed_data(k_vec_len, 0.45f, 0.5f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    const bool setup_ok = finalize_graph(c, ggml_mul(c.ctx, a, b));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[3] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_mul").c_str(), cfg, fn));
  }

  {
    auto lhs = make_positive_data(k_vec_len, 0.3f, 0.25f);
    auto rhs = make_positive_data(k_vec_len, 0.2f, 0.75f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    ggml_tensor * b = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, lhs);
    set_tensor_f32(b, rhs);
    const bool setup_ok = finalize_graph(c, ggml_div(c.ctx, a, b));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[4] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_div").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.5f, 0.125f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_sqr(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[5] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sqr").c_str(), cfg, fn));
  }

  {
    auto src = make_positive_data(k_vec_len, 0.35f, 0.2f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_sqrt(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[6] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sqrt").c_str(), cfg, fn));
  }

  {
    auto src = make_positive_data(k_vec_len, 0.4f, 0.125f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_log(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[7] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_log").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_sin(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[8] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_sin").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.2f, -0.2f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_cos(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[9] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_cos").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_softmax_width * k_softmax_rows, 0.1f, 0.05f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_softmax_width, k_softmax_rows);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_soft_max(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[static_cast<size_t>(k_softmax_width)] : -1.0f;
    };
    results.push_back(emel::bench::measure_case(make_case_name(backend, "op_soft_max").c_str(), cfg, fn));
  }

  {
    auto matrix_b = make_signed_data(k_mm_k * k_mm_m, 0.12f, 0.25f); // [m, k]
    auto matrix_a = make_signed_data(k_mm_k * k_mm_n, 0.08f, -0.1f); // [n, k]
    ggml_graph_case c;

    ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_n); // [n, k]
    ggml_tensor * b = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k_mm_k, k_mm_m); // [m, k]
    set_tensor_f32(a, matrix_a);
    set_tensor_f32(b, matrix_b);

    const bool setup_ok = finalize_graph(c, ggml_mul_mat(c.ctx, a, b));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[0] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_mul_mat").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_neg(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[10] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_neg").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.75f, -0.25f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_relu(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[11] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_relu").c_str(), cfg, fn));
  }

  {
    auto src = make_signed_data(k_vec_len, 0.35f, 0.1f);
    ggml_graph_case c;
    ggml_tensor * a = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, k_vec_len);
    set_tensor_f32(a, src);
    const bool setup_ok = finalize_graph(c, ggml_exp(c.ctx, a));

    auto fn = [&]() {
      const bool ok = setup_ok && c.compute();
      sink += ok ? ggml_get_data_f32(c.out)[12] : -1.0f;
    };
    results.push_back(
      emel::bench::measure_case(make_case_name(backend, "op_unary_exp").c_str(), cfg, fn));
  }

  (void) sink;
}

}  // namespace

namespace emel::bench {

void append_emel_kernel_cases(std::vector<result> & results, const config & cfg) {
  const emel::kernel::x86_64::action::context x86_ctx{};
  const emel::kernel::aarch64::action::context aarch_ctx{};

  auto x86_exec = [&](const auto & ev) {
    return emel::kernel::x86_64::detail::execute_request(ev, x86_ctx);
  };
  auto aarch_exec = [&](const auto & ev) {
    return emel::kernel::aarch64::detail::execute_request(ev, aarch_ctx);
  };

  append_emel_backend_cases(results, cfg, "x86_64", x86_exec);
  append_emel_backend_cases(results, cfg, "aarch64", aarch_exec);
}

void append_reference_kernel_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "x86_64");
  append_reference_backend_cases(results, cfg, "aarch64");
}

}  // namespace emel::bench
