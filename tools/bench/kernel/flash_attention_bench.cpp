#include "bench_cases.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/sm.hpp"

#include "kernel/bench_common.hpp"
#include "ggml-backend.h"

namespace {

constexpr const char * k_flash_env_head_dim = "EMEL_BENCH_FLASH_HEAD_DIM";
constexpr const char * k_flash_env_head_count = "EMEL_BENCH_FLASH_HEAD_COUNT";
constexpr const char * k_flash_env_kv_head_count = "EMEL_BENCH_FLASH_KV_HEAD_COUNT";
constexpr const char * k_flash_env_kv_tokens = "EMEL_BENCH_FLASH_KV_TOKENS";
constexpr const char * k_flash_env_masked_total_tokens = "EMEL_BENCH_FLASH_MASKED_TOTAL_TOKENS";

constexpr uint64_t k_flash_default_head_dim = 64u;
constexpr uint64_t k_flash_default_head_count = 12u;
constexpr uint64_t k_flash_default_kv_head_count = 12u;
constexpr uint64_t k_flash_default_kv_tokens = 104u;
constexpr size_t k_flash_reference_arena_min_bytes = 160u * 1024u * 1024u;
constexpr size_t k_flash_reference_arena_guard_bytes = 8u * 1024u * 1024u;

struct flash_attention_case_spec {
  uint64_t head_dim = k_flash_default_head_dim;
  uint64_t head_count = k_flash_default_head_count;
  uint64_t kv_head_count = k_flash_default_kv_head_count;
  uint64_t kv_tokens = k_flash_default_kv_tokens;
  uint64_t masked_total_tokens = k_flash_default_kv_tokens;
};

struct flash_attention_case_buffers {
  std::vector<float> q_emel = {};
  std::vector<uint16_t> k_emel = {};
  std::vector<uint16_t> v_emel = {};
  std::vector<float> q_reference = {};
  std::vector<float> k_reference = {};
  std::vector<float> v_reference = {};
};

template <class tensor_type>
void fill_dense_nb(tensor_type & tensor, const uint64_t elem_size) {
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_src_view_3d(const void * data,
                                                  const emel::kernel::event::dtype type,
                                                  const uint64_t elem_size,
                                                  const uint64_t ne0,
                                                  const uint64_t ne1,
                                                  const uint64_t ne2) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = type;
  tensor.ne = {ne0, ne1, ne2, 1u};
  fill_dense_nb(tensor, elem_size);
  return tensor;
}

emel::kernel::event::tensor_view make_src_view_strided_3d(const void * data,
                                                          const emel::kernel::event::dtype type,
                                                          const uint64_t elem_size,
                                                          const uint64_t ne0,
                                                          const uint64_t ne1,
                                                          const uint64_t ne2,
                                                          const uint64_t nb1,
                                                          const uint64_t nb2) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = type;
  tensor.ne = {ne0, ne1, ne2, 1u};
  tensor.nb[0] = elem_size;
  tensor.nb[1] = nb1;
  tensor.nb[2] = nb2;
  tensor.nb[3] = nb2 * ne2;
  return tensor;
}

emel::kernel::event::tensor_view_mut make_dst_view_3d(float * data,
                                                      const uint64_t ne0,
                                                      const uint64_t ne1,
                                                      const uint64_t ne2) {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, 1u};
  fill_dense_nb(tensor, sizeof(float));
  return tensor;
}

uint64_t read_env_u64(const char * name, const uint64_t fallback) {
  const char * value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed == 0u) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

flash_attention_case_spec load_flash_attention_case_spec() {
  flash_attention_case_spec spec{};
  spec.head_dim = read_env_u64(k_flash_env_head_dim, spec.head_dim);
  spec.head_count = read_env_u64(k_flash_env_head_count, spec.head_count);
  spec.kv_head_count = read_env_u64(k_flash_env_kv_head_count, spec.kv_head_count);
  spec.kv_tokens = read_env_u64(k_flash_env_kv_tokens, spec.kv_tokens);
  spec.masked_total_tokens =
      read_env_u64(k_flash_env_masked_total_tokens, spec.kv_tokens);
  if (spec.kv_head_count == 0u || spec.head_count == 0u || spec.head_dim == 0u ||
      spec.kv_tokens == 0u || spec.head_count % spec.kv_head_count != 0u ||
      spec.masked_total_tokens < spec.kv_tokens) {
    spec = {};
  }
  return spec;
}

flash_attention_case_buffers make_flash_attention_case_buffers(
    const flash_attention_case_spec & spec) {
  flash_attention_case_buffers buffers{};
  const uint64_t q_count = spec.head_dim * spec.head_count;
  const uint64_t kv_count_emel = spec.head_dim * spec.kv_head_count * spec.kv_tokens;
  const uint64_t kv_count_reference = spec.head_dim * spec.kv_tokens * spec.kv_head_count;

  buffers.q_emel.resize(static_cast<size_t>(q_count));
  buffers.k_emel.resize(static_cast<size_t>(kv_count_emel));
  buffers.v_emel.resize(static_cast<size_t>(kv_count_emel));
  buffers.q_reference.resize(static_cast<size_t>(q_count));
  buffers.k_reference.resize(static_cast<size_t>(kv_count_reference));
  buffers.v_reference.resize(static_cast<size_t>(kv_count_reference));

  for (uint64_t head = 0; head < spec.head_count; ++head) {
    for (uint64_t dim = 0; dim < spec.head_dim; ++dim) {
      const double angle = static_cast<double>((head + 1u) * (dim + 3u));
      const float value = emel::kernel::detail::quant::fp16_to_fp32(
          emel::kernel::detail::quant::fp32_to_fp16(
              static_cast<float>(std::sin(angle * 0.03125))));
      buffers.q_emel[static_cast<size_t>(head * spec.head_dim + dim)] = value;
      buffers.q_reference[static_cast<size_t>(head * spec.head_dim + dim)] = value;
    }
  }

  for (uint64_t token = 0; token < spec.kv_tokens; ++token) {
    for (uint64_t head = 0; head < spec.kv_head_count; ++head) {
      for (uint64_t dim = 0; dim < spec.head_dim; ++dim) {
        const double base = static_cast<double>((token + 1u) * (head + 3u) * (dim + 5u));
        const uint16_t k_bits = emel::kernel::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(base * 0.0078125)));
        const uint16_t v_bits = emel::kernel::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(base * 0.01171875)));
        const float k_value = emel::kernel::detail::quant::fp16_to_fp32(k_bits);
        const float v_value = emel::kernel::detail::quant::fp16_to_fp32(v_bits);
        const uint64_t emel_offset =
            head * spec.kv_tokens * spec.head_dim + token * spec.head_dim + dim;
        const uint64_t reference_offset =
            dim + spec.head_dim * token + spec.head_dim * spec.kv_tokens * head;
        buffers.k_emel[static_cast<size_t>(emel_offset)] = k_bits;
        buffers.v_emel[static_cast<size_t>(emel_offset)] = v_bits;
        buffers.k_reference[static_cast<size_t>(reference_offset)] = k_value;
        buffers.v_reference[static_cast<size_t>(reference_offset)] = v_value;
      }
    }
  }

  return buffers;
}

size_t flash_attention_reference_arena_bytes(const flash_attention_case_spec & spec) {
  const size_t q_bytes =
      static_cast<size_t>(spec.head_dim * spec.head_count) * sizeof(float);
  const size_t kv_bytes =
      static_cast<size_t>(spec.head_dim * spec.kv_tokens * spec.kv_head_count) *
      sizeof(ggml_fp16_t);
  const size_t out_bytes =
      static_cast<size_t>(spec.head_dim * spec.head_count) * sizeof(float);
  const size_t score_bytes =
      static_cast<size_t>(spec.masked_total_tokens * spec.head_count) * sizeof(float);
  const size_t tensor_overhead_bytes = ggml_tensor_overhead() * 8u;
  const size_t graph_overhead_bytes = ggml_graph_overhead_custom(8u, false);
  const size_t working_set_bytes =
      q_bytes + (2u * kv_bytes) + out_bytes + (8u * score_bytes);
  return std::max(k_flash_reference_arena_min_bytes,
                  tensor_overhead_bytes + graph_overhead_bytes + working_set_bytes) +
      k_flash_reference_arena_guard_bytes;
}

emel::kernel::event::op_flash_attn_ext make_flash_attention_event(
    const flash_attention_case_spec & spec,
    const flash_attention_case_buffers & buffers,
    float * dst) {
  emel::kernel::event::op_flash_attn_ext ev{};
  const float scale = 1.0f / std::sqrt(static_cast<float>(spec.head_dim));
  const uint32_t masked_total_tokens = static_cast<uint32_t>(spec.masked_total_tokens);

  ev.src0 = make_src_view_3d(buffers.q_emel.data(),
                             emel::kernel::event::dtype::f32,
                             sizeof(float),
                             spec.head_dim,
                             1u,
                             spec.head_count);
  ev.src1 = make_src_view_strided_3d(
      buffers.k_emel.data(),
      emel::kernel::event::dtype::f16,
      sizeof(uint16_t),
      spec.head_dim,
      spec.kv_tokens,
      spec.kv_head_count,
      sizeof(uint16_t) * spec.head_dim,
      sizeof(uint16_t) * spec.kv_tokens * spec.head_dim);
  ev.src2 = make_src_view_strided_3d(
      buffers.v_emel.data(),
      emel::kernel::event::dtype::f16,
      sizeof(uint16_t),
      spec.head_dim,
      spec.kv_tokens,
      spec.kv_head_count,
      sizeof(uint16_t) * spec.head_dim,
      sizeof(uint16_t) * spec.kv_tokens * spec.head_dim);
  ev.dst = make_dst_view_3d(dst, spec.head_dim, 1u, spec.head_count);
  ev.nth = 1;
  std::memcpy(ev.op_params.data(), &scale, sizeof(scale));
  std::memcpy(ev.op_params.data() + sizeof(scale),
              &masked_total_tokens,
              sizeof(masked_total_tokens));
  ev.op_params_size = sizeof(scale) + sizeof(masked_total_tokens);
  return ev;
}

void set_tensor_f16(ggml_tensor * tensor, const std::vector<float> & values) {
  ggml_fp32_to_fp16_row(values.data(),
                        static_cast<ggml_fp16_t *>(tensor->data),
                        static_cast<int64_t>(values.size()));
}

const char * flash_attention_backend_name() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#else
  return "host";
#endif
}

std::string make_flash_attention_case_name() {
  return std::string{"flash_attention/"} + flash_attention_backend_name() +
      "/op_flash_attn_ext_decode_like";
}

template <class exec_fn>
void append_emel_flash_attention_host_case(std::vector<emel::bench::result> & results,
                                           const emel::bench::config & cfg,
                                           exec_fn exec) {
  volatile float sink = 0.0f;
  const auto spec = load_flash_attention_case_spec();
  const auto buffers = make_flash_attention_case_buffers(spec);
  std::vector<float> dst(static_cast<size_t>(spec.head_dim * spec.head_count), 0.0f);
  const auto ev = make_flash_attention_event(spec, buffers, dst.data());
  const std::string case_name = make_flash_attention_case_name();

  auto fn = [&]() {
    const bool ok = exec(ev);
    sink += ok ? dst[0] : -1.0f;
  };
  results.push_back(emel::bench::measure_case(case_name.c_str(), cfg, fn));
  (void) sink;
}

void append_reference_flash_attention_host_case(std::vector<emel::bench::result> & results,
                                                const emel::bench::config & cfg) {
  volatile float sink = 0.0f;
  const auto spec = load_flash_attention_case_spec();
  const auto buffers = make_flash_attention_case_buffers(spec);
  const float scale = 1.0f / std::sqrt(static_cast<float>(spec.head_dim));
  ggml_graph_case c(flash_attention_reference_arena_bytes(spec));

  ggml_tensor * q = ggml_new_tensor_3d(c.ctx,
                                       GGML_TYPE_F32,
                                       static_cast<int64_t>(spec.head_dim),
                                       1,
                                       static_cast<int64_t>(spec.head_count));
  ggml_tensor * k = ggml_new_tensor_3d(c.ctx,
                                       GGML_TYPE_F16,
                                       static_cast<int64_t>(spec.head_dim),
                                       static_cast<int64_t>(spec.kv_tokens),
                                       static_cast<int64_t>(spec.kv_head_count));
  ggml_tensor * v = ggml_new_tensor_3d(c.ctx,
                                       GGML_TYPE_F16,
                                       static_cast<int64_t>(spec.head_dim),
                                       static_cast<int64_t>(spec.kv_tokens),
                                       static_cast<int64_t>(spec.kv_head_count));
  set_tensor_f32(q, buffers.q_reference);
  set_tensor_f16(k, buffers.k_reference);
  set_tensor_f16(v, buffers.v_reference);

  ggml_tensor * out = ggml_flash_attn_ext(c.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
  ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
  c.out = out;
  c.graph = ggml_new_graph(c.ctx);
  const bool setup_ok = c.out != nullptr && c.graph != nullptr;
  if (setup_ok) {
    ggml_build_forward_expand(c.graph, c.out);
  }
  ggml_backend_t backend = ggml_backend_cpu_init();
  if (backend != nullptr) {
    ggml_backend_cpu_set_n_threads(backend, 1);
  }
  ggml_backend_graph_plan_t plan = nullptr;
  if (setup_ok && backend != nullptr) {
    plan = ggml_backend_graph_plan_create(backend, c.graph);
  }
  const std::string case_name = make_flash_attention_case_name();

  auto fn = [&]() {
    const bool ok = setup_ok && backend != nullptr && plan != nullptr &&
        ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS;
    sink += ok ? ggml_get_data_f32(c.out)[0] : -1.0f;
  };
  results.push_back(emel::bench::measure_case(case_name.c_str(), cfg, fn));
  if (plan != nullptr) {
    ggml_backend_graph_plan_free(backend, plan);
  }
  if (backend != nullptr) {
    ggml_backend_free(backend);
  }
  (void) sink;
}

}  // namespace

namespace emel::bench {

void append_emel_flash_attention_cases(std::vector<result> & results, const config & cfg) {
#if defined(__aarch64__) || defined(_M_ARM64)
  emel::kernel::aarch64::sm machine{};
  auto exec = [&](const auto & ev) {
    return machine.process_event(ev);
  };
  append_emel_flash_attention_host_case(results, cfg, exec);
#elif defined(__x86_64__) || defined(_M_X64)
  emel::kernel::x86_64::sm machine{};
  auto exec = [&](const auto & ev) {
    return machine.process_event(ev);
  };
  append_emel_flash_attention_host_case(results, cfg, exec);
#else
  (void) results;
  (void) cfg;
#endif
}

void append_reference_flash_attention_cases(std::vector<result> & results, const config & cfg) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__x86_64__) || defined(_M_X64)
  append_reference_flash_attention_host_case(results, cfg);
#else
  (void) results;
  (void) cfg;
#endif
}

}  // namespace emel::bench
