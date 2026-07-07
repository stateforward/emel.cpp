#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/sm.hpp"

// Reference lane only: ggml/llama.cpp drives the ggml_* comparison results and
// never touches the EMEL lane, per the split-lane benchmark contract.
extern "C" {
#include "ggml-cpu.h"
#include "ggml.h"
}

// Focused parallel-matmul suite: the EMEL lane forks one logical mul_mat into
// per-lane row-slice events across kernel actors on the warm thread pool (the
// production view-sliced decode/prefill parallel route shape). The reference
// lane carries two baselines: the plain-named cases run the identical full
// event through one EMEL kernel actor serially (fork/join speedup proof), and
// the ggml_* cases run the same logical matmul through ggml's warm threadpool
// at the same core budget (n_threads = k_lanes), so the compare row pairs
// EMEL's inter-actor row slicing against ggml's intra-op thread chunking.
// Operand class: production EMEL weight layout for the EMEL lane and matching
// GGML-native raw tensors for the reference lane. Quantized GEMV EMEL cases
// quantize the RHS vector inside the measured function before dispatching the
// prepared-input route, matching the production decode route cost; the
// reference lane remains ggml's native mul_mat path.

namespace {

using emel::kernel::event::dtype;
using emel::kernel::event::op_mul_mat;
using emel::kernel::event::tensor_view;
using emel::kernel::event::tensor_view_mut;

constexpr size_t k_lanes = 8u;
constexpr size_t k_worker_lanes = k_lanes - 1u;
constexpr int32_t k_dim = 2048;
constexpr int32_t k_gemm_tokens = 8;

using lane_pool = emel::policy::fork_join_lane_pool<k_worker_lanes, 128u, 1048576u>;

uint64_t weight_group_rows(const dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  if (code == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
      code == emel::kernel::detail::dtype_q8_0_x4_bl8) {
    return emel::kernel::detail::quant::Q8_0_X4_ROWS;
  }
  if (code == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
      code == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
      code == emel::kernel::detail::dtype_q6_k_x8 ||
      code == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      code == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) {
    return emel::kernel::detail::quant::Q4_K_X8_ROWS;
  }
  return 1u;
}

uint64_t weight_storage_rows(const dtype type, const uint64_t rows) noexcept {
  const uint64_t group_rows = weight_group_rows(type);
  return (rows + group_rows - 1u) / group_rows;
}

emel::kernel::kernel_kind host_kernel_kind() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return emel::kernel::kernel_kind::aarch64;
#else
  return emel::kernel::kernel_kind::x86_64;
#endif
}

tensor_view make_weight_view(const void * data,
                             const dtype type,
                             const uint64_t row_bytes,
                             const int32_t cols,
                             const int32_t rows) {
  tensor_view view{};
  view.data = data;
  view.type = type;
  view.ne = {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows), 1u, 1u};
  view.nb[0] = type == dtype::f32 ? sizeof(float) : 1u;
  view.nb[1] = row_bytes;
  view.nb[2] = row_bytes * weight_storage_rows(type, static_cast<uint64_t>(rows));
  view.nb[3] = view.nb[2];
  return view;
}

tensor_view make_input_view(const float * data, const int32_t tokens, const int32_t cols) {
  tensor_view view{};
  view.data = data;
  view.type = dtype::f32;
  view.ne = {static_cast<uint64_t>(tokens), static_cast<uint64_t>(cols), 1u, 1u};
  view.nb[0] = sizeof(float);
  view.nb[1] = sizeof(float) * static_cast<uint64_t>(tokens);
  view.nb[2] = view.nb[1] * static_cast<uint64_t>(cols);
  view.nb[3] = view.nb[2];
  return view;
}

tensor_view make_q8_k_vector_view(const void * data, const int32_t cols) {
  const size_t row_bytes = emel::kernel::detail::quantized_row_storage_bytes(
      emel::kernel::detail::dtype_q8_k, static_cast<uint64_t>(cols));
  tensor_view view{};
  view.data = data;
  view.type = dtype::q8_k;
  view.ne = {1u, static_cast<uint64_t>(cols), 1u, 1u};
  view.nb[0] = 1u;
  view.nb[1] = row_bytes;
  view.nb[2] = row_bytes;
  view.nb[3] = row_bytes;
  return view;
}

tensor_view make_q8_0_vector_view(const void * data, const int32_t cols) {
  const size_t row_bytes = emel::kernel::detail::quantized_row_storage_bytes(
      emel::kernel::detail::dtype_q8_0, static_cast<uint64_t>(cols));
  tensor_view view{};
  view.data = data;
  view.type = dtype::q8_0;
  view.ne = {1u, static_cast<uint64_t>(cols), 1u, 1u};
  view.nb[0] = 1u;
  view.nb[1] = row_bytes;
  view.nb[2] = row_bytes;
  view.nb[3] = row_bytes;
  return view;
}

tensor_view_mut make_output_view(float * data, const int32_t tokens, const int32_t rows) {
  tensor_view_mut view{};
  view.data = data;
  view.type = dtype::f32;
  view.ne = {static_cast<uint64_t>(tokens), static_cast<uint64_t>(rows), 1u, 1u};
  view.nb[0] = sizeof(float);
  view.nb[1] = sizeof(float) * static_cast<uint64_t>(tokens);
  view.nb[2] = view.nb[1] * static_cast<uint64_t>(rows);
  view.nb[3] = view.nb[2];
  return view;
}

// Mirror the production group-aligned contiguous row split; k_dim rows divide
// evenly by lane count for every case in this suite.
op_mul_mat make_sliced_event(const op_mul_mat & ev,
                             const uint64_t row_begin,
                             const uint64_t row_count) {
  op_mul_mat sliced = ev;
  const uint64_t group_rows = weight_group_rows(ev.src0.type);
  const uint64_t slice_groups = (row_count + group_rows - 1u) / group_rows;
  sliced.src0.data =
      static_cast<const uint8_t *>(ev.src0.data) +
      (row_begin / group_rows) * ev.src0.nb[1];
  sliced.src0.ne[1] = row_count;
  sliced.src0.nb[2] = ev.src0.nb[1] * slice_groups;
  sliced.src0.nb[3] = sliced.src0.nb[2];
  sliced.dst.data = static_cast<uint8_t *>(ev.dst.data) + row_begin * ev.dst.nb[1];
  sliced.dst.ne[1] = row_count;
  sliced.dst.nb[2] = ev.dst.nb[1] * row_count;
  sliced.dst.nb[3] = sliced.dst.nb[2];
  return sliced;
}

struct lane_fixture {
  std::array<emel::kernel::sm, k_lanes> kernels = {};
  lane_pool pool = {};

  lane_fixture() {
    for (auto & kernel : kernels) {
      kernel.set_kind(host_kernel_kind());
    }
  }
};

struct case_buffers {
  std::vector<uint8_t> weights = {};
  std::vector<float> input = {};
  std::vector<emel::kernel::detail::quant::block_q8_k> input_q8_k = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> input_q8_0 = {};
  std::vector<float> output = {};
  op_mul_mat ev = {};
};

case_buffers make_case(const dtype type, const int32_t tokens) {
  case_buffers buffers;
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  const uint64_t raw_row_bytes =
      type == dtype::f32
          ? sizeof(float) * static_cast<uint64_t>(k_dim)
          : emel::kernel::detail::quantized_row_storage_bytes(
                code, static_cast<uint64_t>(k_dim));
  std::vector<uint8_t> raw_weights(raw_row_bytes * static_cast<size_t>(k_dim), 0u);
  for (size_t idx = 0; idx < raw_weights.size(); ++idx) {
    raw_weights[idx] = static_cast<uint8_t>((idx * 31u + 7u) & 0x3fu);
  }
  if (type == dtype::f32) {
    auto * values = reinterpret_cast<float *>(raw_weights.data());
    const size_t count = raw_weights.size() / sizeof(float);
    for (size_t idx = 0; idx < count; ++idx) {
      values[idx] = 0.25f * static_cast<float>((idx * 13u + 5u) % 17u) - 2.0f;
    }
  }
  dtype storage_type = type;
  uint64_t row_bytes = raw_row_bytes;
  if (type == dtype::q4_k) {
    storage_type = dtype::q4_k_x8_bl8;
    row_bytes = emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(
        static_cast<uint64_t>(k_dim));
    const uint64_t group_count =
        emel::kernel::detail::quant::packed_q4_k_x8_group_count(
            static_cast<uint64_t>(k_dim));
    buffers.weights.assign(static_cast<size_t>(row_bytes * group_count), 0u);
    const bool packed = emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
        reinterpret_cast<const emel::kernel::detail::quant::block_q4_k *>(
            raw_weights.data()),
        static_cast<uint64_t>(k_dim),
        static_cast<uint64_t>(k_dim),
        buffers.weights.data());
    if (!packed) {
      std::fprintf(stderr, "error: parallel matmul q4_k pack failed\n");
      std::abort();
    }
  } else if (type == dtype::q8_0) {
    storage_type = dtype::q8_0_x4_bl8;
    row_bytes = emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(
        static_cast<uint64_t>(k_dim));
    const uint64_t group_count =
        emel::kernel::detail::quant::packed_q8_0_x4_group_count(
            static_cast<uint64_t>(k_dim));
    buffers.weights.assign(static_cast<size_t>(row_bytes * group_count), 0u);
    const bool packed = emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
        reinterpret_cast<const emel::kernel::detail::quant::block_q8_0 *>(
            raw_weights.data()),
        static_cast<uint64_t>(k_dim),
        static_cast<uint64_t>(k_dim),
        buffers.weights.data());
    if (!packed) {
      std::fprintf(stderr, "error: parallel matmul q8_0 pack failed\n");
      std::abort();
    }
  } else {
    buffers.weights = std::move(raw_weights);
  }

  buffers.input.assign(
      static_cast<size_t>(tokens) * static_cast<size_t>(k_dim), 0.0f);
  for (size_t idx = 0; idx < buffers.input.size(); ++idx) {
    buffers.input[idx] = 0.125f * static_cast<float>((idx * 7u + 3u) % 19u) - 1.0f;
  }
  buffers.output.assign(
      static_cast<size_t>(tokens) * static_cast<size_t>(k_dim), 0.0f);

  buffers.ev.src0 =
      make_weight_view(buffers.weights.data(), storage_type, row_bytes, k_dim, k_dim);
  buffers.ev.src1 = make_input_view(buffers.input.data(), tokens, k_dim);
  if (tokens == 1 && (type == dtype::q4_k || type == dtype::q6_k)) {
    buffers.input_q8_k.resize(
        static_cast<size_t>(k_dim / emel::kernel::detail::quant::QK_K));
  }
  if (tokens == 1 && type == dtype::q8_0) {
    buffers.input_q8_0.resize(
        static_cast<size_t>(k_dim / emel::kernel::detail::quant::QK8_0));
  }
  buffers.ev.dst = make_output_view(buffers.output.data(), tokens, k_dim);
  return buffers;
}

struct bench_case_spec {
  const char * name;
  dtype type;
  int32_t tokens;
};

constexpr std::array<bench_case_spec, 5> k_cases = {{
    {"parallel_matmul/gemv_f32", dtype::f32, 1},
    {"parallel_matmul/gemv_q8_0", dtype::q8_0, 1},
    {"parallel_matmul/gemv_q4_k", dtype::q4_k, 1},
    {"parallel_matmul/gemv_q6_k", dtype::q6_k, 1},
    {"parallel_matmul/gemm8_f32", dtype::f32, k_gemm_tokens},
}};

// Same EMEL fork/join work measured under ggml_* names; the reference lane
// answers these with ggml's threadpool so compare rows read EMEL vs llama.cpp.
constexpr std::array<bench_case_spec, 5> k_ggml_cases = {{
    {"parallel_matmul/ggml_gemv_f32", dtype::f32, 1},
    {"parallel_matmul/ggml_gemv_q8_0", dtype::q8_0, 1},
    {"parallel_matmul/ggml_gemv_q4_k", dtype::q4_k, 1},
    {"parallel_matmul/ggml_gemv_q6_k", dtype::q6_k, 1},
    {"parallel_matmul/ggml_gemm8_f32", dtype::f32, k_gemm_tokens},
}};

bool prepare_emel_case_rhs(case_buffers & buffers,
                           const bench_case_spec & spec) noexcept {
  buffers.ev.src1 = make_input_view(buffers.input.data(), spec.tokens, k_dim);
  if (spec.tokens == 1 && (spec.type == dtype::q4_k || spec.type == dtype::q6_k)) {
    if (buffers.input_q8_k.size() !=
        static_cast<size_t>(k_dim / emel::kernel::detail::quant::QK_K)) {
      return false;
    }
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        buffers.input.data(), 1u, buffers.input_q8_k.data(),
        static_cast<int64_t>(k_dim));
    buffers.ev.src1 = make_q8_k_vector_view(buffers.input_q8_k.data(), k_dim);
  }
  if (spec.tokens == 1 && spec.type == dtype::q8_0) {
    if (buffers.input_q8_0.size() !=
        static_cast<size_t>(k_dim / emel::kernel::detail::quant::QK8_0)) {
      return false;
    }
    emel::kernel::detail::quant::quantize_row_q8_0_strided(
        buffers.input.data(), 1u, buffers.input_q8_0.data(),
        static_cast<int64_t>(k_dim));
    buffers.ev.src1 = make_q8_0_vector_view(buffers.input_q8_0.data(), k_dim);
  }
  return true;
}

ggml_type ggml_type_for(const dtype type) {
  switch (type) {
    case dtype::q8_0:
      return GGML_TYPE_Q8_0;
    case dtype::q4_k:
      return GGML_TYPE_Q4_K;
    case dtype::q6_k:
      return GGML_TYPE_Q6_K;
    default:
      return GGML_TYPE_F32;
  }
}

// Reference lane: ggml computes the same logical [k_dim x k_dim] @ [k_dim x
// tokens] matmul as one mul_mat node, parallelized by its own warm threadpool
// over the same core budget as the EMEL lane (n_threads = k_lanes). Weight
// bytes use the identical deterministic pattern as make_case; block layouts
// (q8_0/q4_K/q6_K) are byte-compatible between the two implementations.
struct ggml_matmul_reference {
  ggml_context * ctx = nullptr;
  ggml_cgraph * graph = nullptr;
  ggml_threadpool * threadpool = nullptr;
  std::vector<uint8_t> work = {};
  ggml_cplan plan = {};
  volatile float sink = 0.0f;

  ggml_matmul_reference(const ggml_type type, const int32_t tokens) {
    const size_t row_bytes = ggml_row_size(type, k_dim);
    const size_t weight_bytes = row_bytes * static_cast<size_t>(k_dim);
    ggml_init_params init{};
    init.mem_size = weight_bytes +
                    static_cast<size_t>(tokens) * static_cast<size_t>(k_dim) *
                        2u * sizeof(float) +
                    32u * 1024u * 1024u;
    init.mem_buffer = nullptr;
    init.no_alloc = false;
    ctx = ggml_init(init);
    if (ctx == nullptr) {
      std::fprintf(stderr, "error: parallel matmul ggml reference init failed\n");
      std::abort();
    }
    graph = ggml_new_graph(ctx);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, k_dim, k_dim);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_dim, tokens);
    if (type == GGML_TYPE_F32) {
      auto * values = static_cast<float *>(weights->data);
      const size_t count = weight_bytes / sizeof(float);
      for (size_t idx = 0; idx < count; ++idx) {
        values[idx] = 0.25f * static_cast<float>((idx * 13u + 5u) % 17u) - 2.0f;
      }
    } else {
      auto * bytes = static_cast<uint8_t *>(weights->data);
      for (size_t idx = 0; idx < weight_bytes; ++idx) {
        bytes[idx] = static_cast<uint8_t>((idx * 31u + 7u) & 0x3fu);
      }
    }
    auto * input = static_cast<float *>(activation->data);
    const size_t input_count = static_cast<size_t>(tokens) * static_cast<size_t>(k_dim);
    for (size_t idx = 0; idx < input_count; ++idx) {
      input[idx] = 0.125f * static_cast<float>((idx * 7u + 3u) % 19u) - 1.0f;
    }
    ggml_build_forward_expand(graph, ggml_mul_mat(ctx, weights, activation));
    ggml_threadpool_params tp =
        ggml_threadpool_params_default(static_cast<int32_t>(k_lanes));
    tp.poll = 100;  // warm polling, matching EMEL's warm lane pool
    threadpool = ggml_threadpool_new(&tp);
    plan = ggml_graph_plan(graph, static_cast<int32_t>(k_lanes), threadpool);
    work.resize(plan.work_size != 0u ? plan.work_size : 1u);
    plan.work_data = work.data();
  }

  ~ggml_matmul_reference() {
    if (threadpool != nullptr) {
      ggml_threadpool_free(threadpool);
    }
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }

  ggml_matmul_reference(const ggml_matmul_reference &) = delete;
  ggml_matmul_reference & operator=(const ggml_matmul_reference &) = delete;

  void run() noexcept {
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "error: parallel matmul ggml reference compute failed\n");
      std::abort();
    }
    sink += 1.0f;
  }
};

}  // namespace

namespace emel::bench {

void append_emel_parallel_matmul_cases(std::vector<result> & results, const config & cfg) {
  volatile float sink = 0.0f;

  const auto measure_parallel = [&](const bench_case_spec & spec) {
    lane_fixture fixture;
    auto buffers = make_case(spec.type, spec.tokens);
    std::array<op_mul_mat, k_lanes> lane_events = {};
    constexpr uint64_t rows_per_lane = static_cast<uint64_t>(k_dim) / k_lanes;
    for (size_t lane = 0; lane < k_lanes; ++lane) {
      lane_events[lane] = make_sliced_event(
          buffers.ev, static_cast<uint64_t>(lane) * rows_per_lane, rows_per_lane);
    }

    auto fn = [&]() {
      if (!prepare_emel_case_rhs(buffers, spec)) {
        std::fprintf(stderr, "error: parallel matmul RHS preparation failed\n");
        std::abort();
      }
      for (auto & lane_ev : lane_events) {
        lane_ev.src1 = buffers.ev.src1;
      }
      std::array<bool, k_lanes> lane_ok = {};
      lane_pool::join_group group{};
      bool all_submitted = true;
      for (size_t lane = 1; lane < k_lanes; ++lane) {
        auto & kernel = fixture.kernels[lane];
        const auto & lane_ev = lane_events[lane];
        auto & ok_flag = lane_ok[lane];
        const bool submitted = fixture.pool.try_submit(
            group, [&kernel, &lane_ev, &ok_flag]() noexcept {
          ok_flag = kernel.process_event(lane_ev);
        });
        all_submitted = all_submitted && submitted;
      }
      if (!all_submitted) {
        (void)group.wait();
        std::fprintf(stderr, "error: parallel matmul lane submit failed\n");
        std::abort();
      }
      lane_ok[0] = fixture.kernels[0].process_event(lane_events[0]);
      const bool joined = group.wait();
      if (!joined) {
        std::fprintf(stderr, "error: parallel matmul lane join failed\n");
        std::abort();
      }
      bool all_ok = true;
      for (const bool ok : lane_ok) {
        all_ok = all_ok && ok;
      }
      sink += all_ok ? buffers.output[0] : -1.0f;
    };
    results.push_back(measure_case(spec.name, cfg, fn));
  };

  for (const auto & spec : k_ggml_cases) {
    measure_parallel(spec);
  }
  for (const auto & spec : k_cases) {
    measure_parallel(spec);
  }
}

void append_reference_parallel_matmul_cases(std::vector<result> & results, const config & cfg) {
  static emel::kernel::sm kernel;
  kernel.set_kind(host_kernel_kind());
  volatile float sink = 0.0f;

  for (const auto & spec : k_ggml_cases) {
    auto fixture =
        std::make_unique<ggml_matmul_reference>(ggml_type_for(spec.type), spec.tokens);
    auto fn = [&fixture]() { fixture->run(); };
    results.push_back(measure_case(spec.name, cfg, fn));
  }
  for (const auto & spec : k_cases) {
    auto buffers = make_case(spec.type, spec.tokens);
    auto fn = [&]() {
      if (!prepare_emel_case_rhs(buffers, spec)) {
        std::fprintf(stderr, "error: parallel matmul serial RHS preparation failed\n");
        std::abort();
      }
      const bool ok = kernel.process_event(buffers.ev);
      sink += ok ? buffers.output[0] : -1.0f;
    };
    results.push_back(measure_case(spec.name, cfg, fn));
  }
}

}  // namespace emel::bench
