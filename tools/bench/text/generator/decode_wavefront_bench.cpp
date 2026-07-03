#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/sm.hpp"
#include "emel/text/generator/decode_wavefront/sm.hpp"

// Reference lane only: ggml/llama.cpp drives the comparison result and never
// touches the EMEL lane, per the split-lane benchmark contract.
extern "C" {
#include "ggml-cpu.h"
#include "ggml.h"
}

namespace {

namespace wavefront = emel::text::generator::decode_wavefront;
using execute_t = emel::graph::processor::event::execute;

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool prepare_graph_reuse(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool alloc_graph_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool bind_inputs_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool run_kernel_counting(const execute_t & request, int32_t * err_out) {
  auto * calls = static_cast<int32_t *>(request.compute_ctx);
  *calls += 1;
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

// Realistic per-lane decode compute: one f32 GEMV  y[dim] = W[dim x dim] @ x[dim],
// representative of an autoregressive decode-step projection. Each lane owns its
// own weights/activation buffers, so lanes share no bottleneck and parallel
// dispatch is the right strategy. Both the scalar and wavefront lanes run the
// identical kernel, so the comparison isolates sequential vs parallel dispatch.
struct gemv_work {
  int32_t dim = 0;
  int32_t calls = 0;
  std::vector<float> weights;  // dim*dim
  std::vector<float> input;    // dim
  std::vector<float> output;   // dim
  volatile float sink = 0.0f;

  void init(int32_t dim_in, uint32_t seed) {
    dim = dim_in;
    weights.resize(static_cast<size_t>(dim) * static_cast<size_t>(dim));
    input.resize(static_cast<size_t>(dim));
    output.resize(static_cast<size_t>(dim));
    uint32_t state = seed * 2654435761u + 1u;
    const auto next = [&state]() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return static_cast<float>((state >> 8) & 0xffffu) / 65536.0f - 0.5f;
    };
    for (auto & value : weights) {
      value = next();
    }
    for (auto & value : input) {
      value = next();
    }
  }

  void compute() noexcept {
    const float * __restrict w = weights.data();
    const float * __restrict x = input.data();
    float * __restrict y = output.data();
    constexpr int32_t k_lanes = 8;  // independent accumulators -> SIMD FMA
    for (int32_t row = 0; row < dim; ++row) {
      const float * __restrict w_row = w + static_cast<size_t>(row) * dim;
      float acc[k_lanes] = {0};
      int32_t col = 0;
      for (; col + k_lanes <= dim; col += k_lanes) {
        for (int32_t k = 0; k < k_lanes; ++k) {
          acc[k] += w_row[col + k] * x[col + k];
        }
      }
      float tail = 0.0f;
      for (; col < dim; ++col) {
        tail += w_row[col] * x[col];
      }
      for (int32_t k = 0; k < k_lanes; ++k) {
        tail += acc[k];
      }
      y[row] = tail;
    }
    sink += y[0];
    calls += 1;
  }
};

bool run_kernel_gemv(const execute_t & request, int32_t * err_out) {
  static_cast<gemv_work *>(request.compute_ctx)->compute();
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

int32_t decode_gemv_dim() {
  const char * value = std::getenv("EMEL_BENCH_DECODE_GEMV_DIM");
  if (value == nullptr || value[0] == '\0') {
    return 256;
  }
  const long parsed = std::strtol(value, nullptr, 10);
  if (parsed < 1 || parsed > 8192) {
    return 256;
  }
  return static_cast<int32_t>(parsed);
}

bool extract_outputs_ok(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

struct reserve_callbacks {
  bool done_called = false;
  bool error_called = false;

  static bool on_done(void * owner, const emel::graph::events::reserve_done &) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->done_called = true;
    return true;
  }

  static bool on_error(void * owner, const emel::graph::events::reserve_error &) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->error_called = true;
    return true;
  }
};

struct compute_callbacks {
  bool done_called = false;
  bool error_called = false;

  void reset() noexcept {
    done_called = false;
    error_called = false;
  }

  static bool on_done(void * owner, const emel::graph::events::compute_done &) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->done_called = true;
    return true;
  }

  static bool on_error(void * owner, const emel::graph::events::compute_error &) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->error_called = true;
    return true;
  }
};

struct lifecycle_fixture {
  int32_t leaf_tensor = 11;
  int32_t compute_tensor = 29;
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 2> tensors{{
      {
          .tensor_id = 0,
          .buffer = &leaf_tensor,
          .buffer_bytes = sizeof(leaf_tensor),
          .consumer_refs = 0,
          .is_leaf = true,
      },
      {
          .tensor_id = 1,
          .buffer = &compute_tensor,
          .buffer_bytes = sizeof(compute_tensor),
          .consumer_refs = 1,
          .is_leaf = false,
      },
  }};
  std::array<int32_t, 1> required_ids = {0};
  std::array<int32_t, 1> publish_ids = {1};
  std::array<int32_t, 1> release_ids = {1};
  emel::graph::processor::event::lifecycle_phase phase{
      .required_filled_ids = required_ids.data(),
      .required_filled_count = static_cast<int32_t>(required_ids.size()),
      .publish_ids = publish_ids.data(),
      .publish_count = static_cast<int32_t>(publish_ids.size()),
      .release_ids = release_ids.data(),
      .release_count = static_cast<int32_t>(release_ids.size()),
  };
  emel::graph::processor::event::lifecycle_manifest reserve{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .phase = nullptr,
  };
  emel::graph::processor::event::lifecycle_manifest compute{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .phase = &phase,
  };
};

struct graph_lane_fixture {
  emel::graph::sm graph{};
  lifecycle_fixture lifecycle{};
  reserve_callbacks reserve_cb{};
  compute_callbacks compute_cb{};
  emel::graph::event::reserve_output reserve_output{};
  emel::graph::event::compute_output compute_output{};
  emel::graph::event::compute compute_request{};
  int32_t kernel_calls = 0;
  int32_t gemv_dim = 0;
  uint32_t gemv_seed = 0;
  gemv_work kernel_work{};
  bool lane_accepted = false;

  void reserve_graph() {
    const emel::graph::event::reserve reserve_request{
        .model_topology = reinterpret_cast<const void *>(0xA5),
        .output_out = &reserve_output,
        .lifecycle = &lifecycle.reserve,
        .max_node_count = 4u,
        .max_tensor_count = 5u,
        .bytes_per_tensor = 8u,
        .workspace_capacity_bytes = 64u,
        .dispatch_done = {&reserve_cb, reserve_callbacks::on_done},
        .dispatch_error = {&reserve_cb, reserve_callbacks::on_error},
    };
    if (!graph.process_event(reserve_request) || !reserve_cb.done_called ||
        reserve_cb.error_called) {
      std::fprintf(stderr, "error: decode wavefront bench graph reserve failed\n");
      std::abort();
    }
  }

  void bind_compute() {
    void * kernel_ctx = &kernel_calls;
    bool (*kernel)(const execute_t &, int32_t *) = run_kernel_counting;
    if (gemv_dim > 0) {
      kernel_work.init(gemv_dim, gemv_seed);
      kernel_ctx = &kernel_work;
      kernel = run_kernel_gemv;
    }
    compute_request = emel::graph::event::compute{
        .step_plan = reinterpret_cast<const void *>(0xB6),
        .output_out = &compute_output,
        .lifecycle = &lifecycle.compute,
        .node_count_hint = reserve_output.node_count,
        .tensor_count_hint = reserve_output.tensor_count,
        .bytes_per_tensor = 8u,
        .workspace_capacity_bytes = 64u,
        .step_index = 0,
        .step_size = 1,
        .kv_tokens = 1,
        .expected_outputs = 1,
        .compute_ctx = kernel_ctx,
        .validate = validate_ok,
        .prepare_graph = prepare_graph_reuse,
        .alloc_graph = alloc_graph_ok,
        .bind_inputs = bind_inputs_ok,
        .run_kernel = kernel,
        .extract_outputs = extract_outputs_ok,
        .dispatch_done = {&compute_cb, compute_callbacks::on_done},
        .dispatch_error = {&compute_cb, compute_callbacks::on_error},
    };
  }

  void reset_iteration() noexcept {
    compute_output = {};
    compute_cb.reset();
    lane_accepted = false;
  }
};

wavefront::event::compatibility_key make_key(const void * model_identity,
                                             const void * backend_identity) {
  return wavefront::event::compatibility_key{
      .model_identity = model_identity,
      .backend_identity = backend_identity,
      .kernel_kind = emel::kernel::kernel_kind::x86_64,
      .attention = emel::text::generator::attention_mode::flash,
      .route = wavefront::event::kernel_route::q8_k,
      .output = wavefront::event::output_contract::preselected_argmax,
      .dtype_layout_contract =
          static_cast<uint32_t>(wavefront::event::kernel_route::q8_k),
      .quantized_contract =
          static_cast<uint32_t>(wavefront::event::kernel_route::q8_k),
      .step_size = 1,
      .token_count = 1,
  };
}

template <size_t lane_count>
struct decode_wavefront_fixture {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<std::unique_ptr<graph_lane_fixture>, lane_count> lanes{};
  std::vector<wavefront::event::lane> wavefront_lanes{};
  wavefront::lane_pool pool{};
  wavefront::sm machine{pool};
  volatile int32_t sink = 0;

  explicit decode_wavefront_fixture(int32_t gemv_dim = 0) {
    const auto key = make_key(&model_tag, &backend_tag);
    wavefront_lanes.reserve(lane_count);
    for (size_t lane_index = 0u; lane_index < lane_count; ++lane_index) {
      lanes[lane_index] = std::make_unique<graph_lane_fixture>();
      lanes[lane_index]->gemv_dim = gemv_dim;
      lanes[lane_index]->gemv_seed =
          static_cast<uint32_t>(lane_index * 131u + 17u);
      lanes[lane_index]->reserve_graph();
      lanes[lane_index]->bind_compute();
      wavefront_lanes.emplace_back(lanes[lane_index]->graph,
                                   lanes[lane_index]->compute_request,
                                   key,
                                   lanes[lane_index]->lane_accepted);
    }
  }

  void run_scalar() noexcept {
    int32_t accepted_count = 0;
    for (auto & lane : lanes) {
      lane->reset_iteration();
      const emel::graph::event::compute_reserved reserved_compute{lane->compute_request};
      const bool accepted = lane->graph.process_event(reserved_compute);
      if (!accepted || !lane->compute_cb.done_called || lane->compute_cb.error_called) {
        std::fprintf(stderr, "error: decode wavefront bench reserved scalar failed\n");
        std::abort();
      }
      accepted_count += static_cast<int32_t>(accepted);
    }
    sink += accepted_count;
  }

  void run_wavefront() noexcept {
    for (auto & lane : lanes) {
      lane->reset_iteration();
    }
    wavefront::event::dispatch_summary summary{};
    wavefront::event::run request{std::span<wavefront::event::lane>{wavefront_lanes},
                                  summary};
    const bool accepted = machine.process_event(request);
    if (!accepted || summary.dispatched_lanes != static_cast<int32_t>(lane_count)) {
      std::fprintf(stderr,
                   "error: decode wavefront bench wavefront dispatch failed "
                   "lanes=%zu accepted=%d err=%d dispatched=%d failed_lane=%d\n",
                   lane_count,
                   accepted ? 1 : 0,
                   summary.err,
                   summary.dispatched_lanes,
                   summary.failed_lane);
      std::abort();
    }
    for (const auto & lane : lanes) {
      if (!lane->lane_accepted || !lane->compute_cb.done_called ||
          lane->compute_cb.error_called) {
        std::fprintf(stderr, "error: decode wavefront bench wavefront lane failed\n");
        std::abort();
      }
    }
    sink += accepted ? summary.dispatched_lanes : -1;
  }
};

template <size_t lane_count>
constexpr const char * case_name() noexcept;

template <>
constexpr const char * case_name<1>() noexcept {
  return "decode_wavefront/batch1";
}

template <>
constexpr const char * case_name<4>() noexcept {
  return "decode_wavefront/batch4";
}

template <>
constexpr const char * case_name<8>() noexcept {
  return "decode_wavefront/batch8";
}

template <size_t lane_count>
emel::bench::result annotate_result(emel::bench::result result,
                                    const std::string_view lane_name) {
  result.lane = std::string{lane_name};
  result.backend_id = lane_name == "wavefront" ? "emel_decode_wavefront_sm"
                                               : "emel_graph_sm_reserved_scalar";
  result.workload_id = "reserved_decode_graph_dispatch";
  result.comparison_mode = "reserved_scalar_vs_wavefront";
  result.output_tokens = lane_count;
  result.max_output_tokens = lane_count;
  result.comparable = true;
  result.note = "reserved graph compute fixture; scalar lane is direct per-lane reserved compute";
  return result;
}

template <size_t lane_count>
void append_wavefront_case(std::vector<emel::bench::result> & results,
                           const emel::bench::config & cfg) {
  auto fixture = std::make_unique<decode_wavefront_fixture<lane_count>>();
  auto fn = [&fixture]() { fixture->run_wavefront(); };
  results.push_back(annotate_result<lane_count>(
      emel::bench::measure_case(case_name<lane_count>(), cfg, fn), "wavefront"));
}

template <size_t lane_count>
void append_scalar_case(std::vector<emel::bench::result> & results,
                        const emel::bench::config & cfg) {
  auto fixture = std::make_unique<decode_wavefront_fixture<lane_count>>();
  auto fn = [&fixture]() { fixture->run_scalar(); };
  results.push_back(annotate_result<lane_count>(
      emel::bench::measure_case(case_name<lane_count>(), cfg, fn), "reserved_scalar"));
}

template <size_t lane_count>
constexpr const char * gemv_case_name() noexcept;

template <>
constexpr const char * gemv_case_name<1>() noexcept {
  return "decode_wavefront/gemv_batch1";
}

template <>
constexpr const char * gemv_case_name<4>() noexcept {
  return "decode_wavefront/gemv_batch4";
}

template <>
constexpr const char * gemv_case_name<8>() noexcept {
  return "decode_wavefront/gemv_batch8";
}

// Realistic-compute variant: each lane runs a decode-representative GEMV, so the
// comparison reflects real per-token decode cost. The wavefront lane forks the
// lanes across the thread-pool co_sm; the reserved-scalar lane runs them
// sequentially through one graph sm. Both run the identical kernel.
template <size_t lane_count>
void append_gemv_wavefront_case(std::vector<emel::bench::result> & results,
                                const emel::bench::config & cfg) {
  auto fixture =
      std::make_unique<decode_wavefront_fixture<lane_count>>(decode_gemv_dim());
  auto fn = [&fixture]() { fixture->run_wavefront(); };
  results.push_back(annotate_result<lane_count>(
      emel::bench::measure_case(gemv_case_name<lane_count>(), cfg, fn), "wavefront"));
}

template <size_t lane_count>
void append_gemv_scalar_case(std::vector<emel::bench::result> & results,
                             const emel::bench::config & cfg) {
  auto fixture =
      std::make_unique<decode_wavefront_fixture<lane_count>>(decode_gemv_dim());
  auto fn = [&fixture]() { fixture->run_scalar(); };
  results.push_back(annotate_result<lane_count>(
      emel::bench::measure_case(gemv_case_name<lane_count>(), cfg, fn), "reserved_scalar"));
}

template <size_t lane_count>
constexpr const char * ggml_case_name() noexcept;

template <>
constexpr const char * ggml_case_name<1>() noexcept {
  return "decode_wavefront/ggml_batch1";
}

template <>
constexpr const char * ggml_case_name<4>() noexcept {
  return "decode_wavefront/ggml_batch4";
}

template <>
constexpr const char * ggml_case_name<8>() noexcept {
  return "decode_wavefront/ggml_batch8";
}

// Reference lane: ggml/llama.cpp computes lane_count independent f32 GEMVs as one
// graph of mul_mat nodes, parallelized by its own warm threadpool over the same
// core budget (n_threads = lane_count). This mirrors the EMEL wavefront's work
// (lane_count independent f32 [dim x dim] @ [dim] projections) but with ggml's
// intra-op threading instead of EMEL's inter-op fork/join. EMEL-owned code is
// never invoked here.
struct ggml_decode_reference {
  int32_t dim = 0;
  int32_t n_threads = 0;
  ggml_context * ctx = nullptr;
  ggml_cgraph * graph = nullptr;
  ggml_threadpool * threadpool = nullptr;
  std::vector<uint8_t> work{};
  ggml_cplan plan{};
  volatile float sink = 0.0f;

  ggml_decode_reference(int32_t lanes, int32_t dim_in)
      : dim(dim_in), n_threads(lanes) {
    const size_t arena =
        static_cast<size_t>(lanes) *
            (static_cast<size_t>(dim) * dim + 2u * dim) * sizeof(float) +
        32u * 1024u * 1024u;
    ggml_init_params init{};
    init.mem_size = arena;
    init.mem_buffer = nullptr;
    init.no_alloc = false;
    ctx = ggml_init(init);
    if (ctx == nullptr) {
      std::fprintf(stderr, "error: decode wavefront ggml reference init failed\n");
      std::abort();
    }
    graph = ggml_new_graph(ctx);
    for (int32_t lane = 0; lane < lanes; ++lane) {
      ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
      ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
      uint32_t state = static_cast<uint32_t>(lane) * 131u + 17u;
      state = state * 2654435761u + 1u;
      const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<float>((state >> 8) & 0xffffu) / 65536.0f - 0.5f;
      };
      float * wd = static_cast<float *>(weights->data);
      for (size_t i = 0; i < static_cast<size_t>(dim) * dim; ++i) {
        wd[i] = next();
      }
      float * xd = static_cast<float *>(activation->data);
      for (int32_t i = 0; i < dim; ++i) {
        xd[i] = next();
      }
      ggml_build_forward_expand(graph, ggml_mul_mat(ctx, weights, activation));
    }
    ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads);
    tp.poll = 100;  // warm polling, matching EMEL's warm worker pool
    threadpool = ggml_threadpool_new(&tp);
    plan = ggml_graph_plan(graph, n_threads, threadpool);
    work.resize(plan.work_size != 0u ? plan.work_size : 1u);
    plan.work_data = work.data();
  }

  ~ggml_decode_reference() {
    if (threadpool != nullptr) {
      ggml_threadpool_free(threadpool);
    }
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }

  ggml_decode_reference(const ggml_decode_reference &) = delete;
  ggml_decode_reference & operator=(const ggml_decode_reference &) = delete;

  void run() noexcept {
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "error: decode wavefront ggml reference compute failed\n");
      std::abort();
    }
    sink += 1.0f;
  }
};

template <size_t lane_count>
void append_ggml_wavefront_case(std::vector<emel::bench::result> & results,
                                const emel::bench::config & cfg) {
  auto fixture =
      std::make_unique<decode_wavefront_fixture<lane_count>>(decode_gemv_dim());
  auto fn = [&fixture]() { fixture->run_wavefront(); };
  results.push_back(annotate_result<lane_count>(
      emel::bench::measure_case(ggml_case_name<lane_count>(), cfg, fn), "wavefront"));
}

template <size_t lane_count>
void append_ggml_reference_case(std::vector<emel::bench::result> & results,
                                const emel::bench::config & cfg) {
  auto fixture = std::make_unique<ggml_decode_reference>(
      static_cast<int32_t>(lane_count), decode_gemv_dim());
  auto fn = [&fixture]() { fixture->run(); };
  auto result = emel::bench::measure_case(ggml_case_name<lane_count>(), cfg, fn);
  result.lane = "ggml";
  result.backend_id = "ggml_threadpool_mul_mat";
  result.workload_id = "independent_decode_gemv_lanes";
  result.comparison_mode = "wavefront_vs_ggml";
  result.output_tokens = lane_count;
  result.max_output_tokens = lane_count;
  result.comparable = true;
  result.note = "ggml reference: lane_count independent f32 GEMV mul_mat, warm threadpool";
  results.push_back(std::move(result));
}

}  // namespace

namespace emel::bench {

void append_emel_decode_wavefront_cases(std::vector<result> & results,
                                        const config & cfg) {
  append_wavefront_case<1>(results, cfg);
  append_wavefront_case<4>(results, cfg);
  append_wavefront_case<8>(results, cfg);
  append_gemv_wavefront_case<1>(results, cfg);
  append_gemv_wavefront_case<4>(results, cfg);
  append_gemv_wavefront_case<8>(results, cfg);
  append_ggml_wavefront_case<1>(results, cfg);
  append_ggml_wavefront_case<4>(results, cfg);
  append_ggml_wavefront_case<8>(results, cfg);
}

void append_reference_decode_wavefront_cases(std::vector<result> & results,
                                             const config & cfg) {
  append_scalar_case<1>(results, cfg);
  append_scalar_case<4>(results, cfg);
  append_scalar_case<8>(results, cfg);
  append_gemv_scalar_case<1>(results, cfg);
  append_gemv_scalar_case<4>(results, cfg);
  append_gemv_scalar_case<8>(results, cfg);
  append_ggml_reference_case<1>(results, cfg);
  append_ggml_reference_case<4>(results, cfg);
  append_ggml_reference_case<8>(results, cfg);
}

}  // namespace emel::bench
