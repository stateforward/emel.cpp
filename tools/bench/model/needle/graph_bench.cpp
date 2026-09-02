// Needle graph serial/parallel4 decode and prefill benchmarks on the pinned
// tests/models/route-w4-qat.cact fixture. Every measured route drives its own
// graph machine ONLY through public events (init/prefill/decode) via the
// maintained loader chain (cact loader probe/bind/parse -> needle binder ->
// graph machine).
//
// Reference lane: the closed-source libneedle 2.0.3 engine cannot be linked;
// its rows are documented constants recorded from the user-supplied
// single-core measurement on this same host class (task contract
// cact-bench-closeout-impl pins 145 decode / 180 prefill tokens/s; the
// training REPORT.md 2026-08-30 measured 132 decode / 180 prefill tps at
// ~700-token context with NEEDLE_THREADS=1). Rows in both lanes carry
// proof_status=measurement_only until snapshot baselines are approved.
#include "bench_cases.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <chrono>
#include <string>
#include <vector>

#include "emel/cact/loader/sm.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/model/needle/sm.hpp"

namespace {

namespace cact_loader = emel::cact::loader;
namespace needle = emel::model::needle;

constexpr char k_serial_decode_case_name[] =
    "needle/graph/decode_steady_route_w4_qat_serial";
constexpr char k_parallel4_decode_case_name[] =
    "needle/graph/decode_steady_route_w4_qat_parallel4";
constexpr char k_serial_prefill_case_name[] =
    "needle/graph/prefill_512_route_w4_qat_serial";
constexpr char k_parallel4_prefill_case_name[] =
    "needle/graph/prefill_512_route_w4_qat_parallel4";
constexpr char k_swa_scalar_exp_128_case_name[] =
    "needle/swa/attend_gqa2_scalar_exp_span128";
constexpr char k_swa_vector_exp_128_case_name[] =
    "needle/swa/attend_gqa2_vector_exp_span128";
constexpr char k_swa_scalar_exp_512_case_name[] =
    "needle/swa/attend_gqa2_scalar_exp_span512";
constexpr char k_swa_vector_exp_512_case_name[] =
    "needle/swa/attend_gqa2_vector_exp_span512";
constexpr char k_swa_scalar_exp_704_case_name[] =
    "needle/swa/attend_gqa2_scalar_exp_span704";
constexpr char k_swa_vector_exp_704_case_name[] =
    "needle/swa/attend_gqa2_vector_exp_span704";
constexpr char k_fwht_case_name[] = "needle/cq/fwht128_avx2";
constexpr char k_hadamard_scalar_case_name[] =
    "needle/hadamard/mlp512_scalar";
constexpr char k_hadamard_avx2_case_name[] = "needle/hadamard/mlp512_avx2";
constexpr char k_hadamard_iters_env[] = "EMEL_BENCH_NEEDLE_HADAMARD_ITERS";
constexpr char k_fwht_iters_env[] = "EMEL_BENCH_NEEDLE_FWHT_ITERS";
constexpr char k_model_env[] = "EMEL_BENCH_NEEDLE_MODEL";
constexpr char k_model_relative_path[] = "tests/models/route-w4-qat.cact";
constexpr char k_model_id[] = "route_w4_qat_cact";
constexpr char k_workload_id[] = "needle_graph_serial_parallel_same_binary_v1";
constexpr char k_decode_iters_env[] = "EMEL_BENCH_NEEDLE_GRAPH_DECODE_ITERS";
constexpr char k_prefill_iters_env[] = "EMEL_BENCH_NEEDLE_GRAPH_PREFILL_ITERS";
constexpr char k_swa_iters_env[] = "EMEL_BENCH_NEEDLE_SWA_ITERS";
constexpr char k_instrument_cq_env[] = "EMEL_BENCH_NEEDLE_GRAPH_INSTRUMENT_CQ";
constexpr char k_instrument_graph_env[] =
    "EMEL_BENCH_NEEDLE_GRAPH_INSTRUMENT_COMPONENTS";

// Steady-state decode is measured after a 128-token prefill; the prefill case
// runs a 512-token prompt (max_seq_len 2048 bounded).
constexpr uint32_t k_decode_context_tokens = 128u;
constexpr uint32_t k_prefill_case_tokens = 512u;

// Documented libneedle reference lane constants (see file header note).
constexpr double k_libneedle_decode_tokens_per_second = 145.0;
constexpr double k_libneedle_prefill_tokens_per_second = 180.0;

constexpr char k_measurement_note[] =
    "proof_status=measurement_only "
    "reference=libneedle_2.0.3_recorded "
    "source=user_supplied_single_core_same_host_class "
    "target_decode_tps=435";

constexpr char k_internal_microbenchmark_note[] =
    "proof_status=measurement_only "
    "comparison_mode=emel_internal_microbenchmark";

std::uint64_t read_env_u64_or(const char *name,
                              const std::uint64_t fallback) noexcept {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || parsed == 0u) {
    return fallback;
  }
  return static_cast<std::uint64_t>(parsed);
}

bool instrument_cq() noexcept {
  const char *value = std::getenv(k_instrument_cq_env);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool instrument_graph() noexcept {
  const char *value = std::getenv(k_instrument_graph_env);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

[[noreturn]] void fail_needle_setup(const char *stage);
emel::bench::result with_needle_metadata(emel::bench::result out,
                                         const char *lane,
                                         const char *backend_id,
                                         const char *backend_language,
                                         std::uint64_t output_tokens);
std::uint64_t benchmark_timestamp_now_ns() noexcept;
emel::bench::result with_internal_microbenchmark_metadata(
    emel::bench::result out, const char *backend_id);

volatile float g_swa_output_sink = 0.0f;
volatile float g_hadamard_output_sink = 0.0f;

void append_hadamard_case(std::vector<emel::bench::result> &results,
                          const emel::bench::config &cfg, const char *name,
                          const bool avx2) {
  constexpr uint32_t n = 512u;
  std::array<float, n> input{};
  std::array<float, n> skip{};
  std::array<float, n> workspace{};
  std::array<float, n> output{};
  std::array<uint16_t, n> d1_bits{};
  std::array<uint16_t, n> d2_bits{};
  std::array<uint16_t, n> d3_bits{};
  for (uint32_t i = 0u; i < n; ++i) {
    input[i] = static_cast<float>(static_cast<int32_t>((i * 37u) % 101u) - 50) *
               0.03125f;
    skip[i] = static_cast<float>(static_cast<int32_t>((i * 19u) % 83u) - 41) *
              0.015625f;
    d1_bits[i] = emel::kernel::detail::quant::fp32_to_fp16(
        static_cast<float>(static_cast<int32_t>((i * 13u) % 29u) - 14) *
        0.125f);
    d2_bits[i] = emel::kernel::detail::quant::fp32_to_fp16(
        static_cast<float>(static_cast<int32_t>((i * 17u) % 31u) - 15) *
        0.09375f);
    d3_bits[i] = emel::kernel::detail::quant::fp32_to_fp16(
        static_cast<float>(static_cast<int32_t>((i * 23u) % 37u) - 18) *
        0.0625f);
  }
  const auto bytes = [](const auto &values) {
    return std::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(values.data()),
        values.size() * sizeof(values[0])};
  };
  const emel::kernel::hadamard::event::mlp_row_request request{
      input, skip, bytes(d1_bits), bytes(d2_bits), bytes(d3_bits), n, n,
      workspace, output};
  emel::kernel::hadamard::sm machine;
  emel::kernel::hadamard::event::dispatch_result dispatch_result{};
  emel::bench::config hadamard_cfg = cfg;
  hadamard_cfg.iterations = read_env_u64_or(k_hadamard_iters_env, 4096u);
  hadamard_cfg.runs = cfg.runs;
  hadamard_cfg.warmup_iterations = 64u;
  hadamard_cfg.warmup_runs = 1u;
  auto fn = [&]() {
    const bool ok = avx2
                        ? machine.process_event(
                              emel::kernel::hadamard::event::execute_mlp_row_avx2{
                                  request, dispatch_result})
                        : machine.process_event(
                              emel::kernel::hadamard::event::execute_mlp_row{
                                  request, dispatch_result});
    if (!ok)
      fail_needle_setup("hadamard_direct");
    g_hadamard_output_sink = output[0];
  };
  results.push_back(with_internal_microbenchmark_metadata(
      emel::bench::measure_case(name, hadamard_cfg, fn),
      avx2 ? "emel_hadamard_avx2" : "emel_hadamard_scalar"));
}

enum class swa_exp_route : uint8_t { scalar = 0u, vector = 1u };

void append_swa_case(std::vector<emel::bench::result> &results,
                     const emel::bench::config &cfg, const char *name,
                     const uint32_t span_len, const swa_exp_route route) {
  constexpr uint32_t capacity = 704u;
  constexpr uint32_t heads = 8u;
  constexpr uint32_t kv_heads = 4u;
  constexpr uint32_t head_dim = 64u;
  std::vector<float> query(static_cast<size_t>(heads) * head_dim);
  std::vector<float> keys(static_cast<size_t>(kv_heads) * capacity * head_dim);
  std::vector<float> values(keys.size());
  std::vector<float> workspace(static_cast<size_t>(span_len) * 2u);
  std::vector<float> output(static_cast<size_t>(heads) * head_dim);
  for (size_t i = 0u; i < query.size(); ++i)
    query[i] = static_cast<float>(static_cast<int32_t>(i % 53u) - 26) *
               0.03125f;
  for (size_t i = 0u; i < keys.size(); ++i) {
    keys[i] = static_cast<float>(static_cast<int32_t>(i % 79u) - 39) *
              0.015625f;
    values[i] = static_cast<float>(static_cast<int32_t>(i % 97u) - 48) *
                0.0078125f;
  }
  const emel::kernel::swa::event::attend_request request{
      .query = query, .key_cache = keys, .value_cache = values,
      .position = span_len - 1u, .window_begin = 0u, .capacity = capacity,
      .heads = heads, .kv_heads = kv_heads, .head_dim = head_dim,
      .workspace = workspace, .output = output};
  emel::kernel::swa::sm machine;
  emel::kernel::swa::event::dispatch_result dispatch_result{};
  emel::bench::config swa_cfg = cfg;
  swa_cfg.iterations = read_env_u64_or(k_swa_iters_env, 256u);
  swa_cfg.runs = cfg.runs;
  swa_cfg.warmup_iterations = 8u;
  swa_cfg.warmup_runs = 1u;
  auto fn = [&]() {
    const bool ok = route == swa_exp_route::vector
                        ? machine.process_event(
                              emel::kernel::swa::event::
                                  execute_attend_gqa2_avx2_vector_exp{
                                      request, dispatch_result})
                        : machine.process_event(
                              emel::kernel::swa::event::execute_attend_gqa2_avx2{
                                  request, dispatch_result});
    if (!ok) fail_needle_setup("swa_direct");
    g_swa_output_sink = output[0];
  };
  results.push_back(with_internal_microbenchmark_metadata(
      emel::bench::measure_case(name, swa_cfg, fn),
      route == swa_exp_route::vector ? "emel_swa_gqa2_avx2_vector_exp"
                                     : "emel_swa_gqa2_avx2_scalar_exp"));
}

void profile_swa_case(const uint32_t span_len, const swa_exp_route route) {
  constexpr uint32_t calls = 1024u;
  emel::bench::config profile_cfg{};
  profile_cfg.iterations = calls;
  profile_cfg.runs = 5u;
  profile_cfg.warmup_iterations = 8u;
  profile_cfg.warmup_runs = 1u;
  const auto measure_pass = [&](const char *pass, auto &&fn) {
    const auto measured = emel::bench::measure_case(pass, profile_cfg, fn);
    std::fprintf(stderr,
                 "# needle_swa_component: route=%s pass=%s "
                 "ns_per_call=%.3f\n",
                 route == swa_exp_route::vector ? "vector_exp" : "scalar_exp",
                 pass, measured.ns_per_op);
  };
  std::vector<float> scores(span_len);
  for (uint32_t i = 0u; i < span_len; ++i)
    scores[i] = -static_cast<float>(i % 101u) * 0.03125f;
  std::vector<float> working(scores.size());
  float sum_sink = 0.0f;
  measure_pass("exp_sum", [&]() {
    std::copy(scores.begin(), scores.end(), working.begin());
    float sum = 0.0f;
    if (route == swa_exp_route::vector) {
      sum = emel::kernel::swa::detail::exp_sum_avx2(
          working.data(), span_len, 0.0f);
    } else {
      for (uint32_t i = 0u; i < span_len; ++i) {
        const float weight = std::exp(working[i]);
        working[i] = weight;
        sum += weight;
      }
    }
    sum_sink = sum;
  });
  g_swa_output_sink = sum_sink;
}

std::uint64_t benchmark_timestamp_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

[[noreturn]] void fail_needle_setup(const char *stage) {
  std::fprintf(stderr, "error: needle_graph bench setup failed: %s\n", stage);
  std::exit(1);
}

void on_loader_probe_done(const cact_loader::events::probe_done &) noexcept {}
void on_loader_probe_error(const cact_loader::events::probe_error &) noexcept {}
void on_loader_bind_done(const cact_loader::events::bind_done &) noexcept {}
void on_loader_bind_error(const cact_loader::events::bind_error &) noexcept {}
void on_loader_parse_done(const cact_loader::events::parse_done &) noexcept {}
void on_loader_parse_error(const cact_loader::events::parse_error &) noexcept {}
void on_needle_bind_done(const needle::events::bind_done &) noexcept {}
void on_needle_bind_error(const needle::events::bind_error &) noexcept {}

const cact_loader::event::probe_done_fn k_probe_done =
    cact_loader::event::probe_done_fn::from<&on_loader_probe_done>();
const cact_loader::event::probe_error_fn k_probe_error =
    cact_loader::event::probe_error_fn::from<&on_loader_probe_error>();
const cact_loader::event::bind_done_fn k_bind_done =
    cact_loader::event::bind_done_fn::from<&on_loader_bind_done>();
const cact_loader::event::bind_error_fn k_bind_error =
    cact_loader::event::bind_error_fn::from<&on_loader_bind_error>();
const cact_loader::event::parse_done_fn k_parse_done =
    cact_loader::event::parse_done_fn::from<&on_loader_parse_done>();
const cact_loader::event::parse_error_fn k_parse_error =
    cact_loader::event::parse_error_fn::from<&on_loader_parse_error>();
const needle::event::bind_done_fn k_needle_done =
    needle::event::bind_done_fn::from<&on_needle_bind_done>();
const needle::event::bind_error_fn k_needle_error =
    needle::event::bind_error_fn::from<&on_needle_bind_error>();

std::filesystem::path resolve_model_path() {
  const char *override_path = std::getenv(k_model_env);
  if (override_path != nullptr && override_path[0] != '\0') {
    return std::filesystem::path{override_path};
  }
#ifdef EMEL_BENCH_REPO_ROOT
  return std::filesystem::path{EMEL_BENCH_REPO_ROOT} / k_model_relative_path;
#else
  return std::filesystem::path{k_model_relative_path};
#endif
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    fail_needle_setup("open_model");
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  if (size <= 0) {
    fail_needle_setup("model_size");
  }
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input.good()) {
    fail_needle_setup("read_model");
  }
  return bytes;
}

// Owns the mmap-equivalent file image, the bound contract, and the graph
// machine for the whole suite run. All graph storage is allocated at machine
// construction; the timed lambdas dispatch public events only.
struct graph_fixture {
  std::vector<uint8_t> file_bytes;
  std::vector<cact_loader::tensor_view> tensors;
  needle::contract contract = {};
  std::unique_ptr<needle::graph::sm> vector_graph;
  std::unique_ptr<needle::graph::scalar_exp_sm> scalar_graph;
  std::vector<float> logits;
  std::vector<int32_t> context_ids;
  std::vector<int32_t> prompt_ids;
  uint32_t decoded_steps = 0u;
  bool swa_vector_exp = true;
  std::uint64_t decode_ns = 0u;
  emel::kernel::cq::event::timing_breakdown cq_timing = {};
  needle::graph::event::timing_breakdown graph_timing = {};

  graph_fixture() : file_bytes(read_file_bytes(resolve_model_path())) {
    cact_loader::sm loader{};
    cact_loader::geometry geometry = {};
    if (!loader.process_event(
            cact_loader::event::probe{std::span<const uint8_t>{file_bytes},
                                      geometry, k_probe_done, k_probe_error})) {
      fail_needle_setup("loader_probe");
    }
    tensors.resize(geometry.num_tensors);
    if (!loader.process_event(cact_loader::event::bind_storage{
            std::span<cact_loader::tensor_view>{tensors}, k_bind_done,
            k_bind_error})) {
      fail_needle_setup("loader_bind_storage");
    }
    if (!loader.process_event(
            cact_loader::event::parse{std::span<const uint8_t>{file_bytes},
                                      k_parse_done, k_parse_error})) {
      fail_needle_setup("loader_parse");
    }

    needle::sm binder{};
    if (!binder.process_event(needle::event::bind{
            geometry, std::span<const cact_loader::tensor_view>{tensors},
            contract, k_needle_done, k_needle_error})) {
      fail_needle_setup("needle_bind");
    }

    vector_graph = std::make_unique<needle::graph::sm>(contract);
    scalar_graph = std::make_unique<needle::graph::scalar_exp_sm>(contract);
    logits.resize(contract.geo.vocab_size);
    context_ids.resize(k_decode_context_tokens);
    prompt_ids.resize(k_prefill_case_tokens);
    // Deterministic pseudo-random in-vocab token stream; the timed path is
    // shape-driven, not content-driven.
    for (size_t i = 0; i < context_ids.size(); ++i) {
      context_ids[i] =
          static_cast<int32_t>((1000003u * i + 7u) % contract.geo.vocab_size);
    }
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
      prompt_ids[i] =
          static_cast<int32_t>((1000033u * i + 13u) % contract.geo.vocab_size);
    }
  }

  template <class event_type> bool process_event(const event_type &ev) {
    return swa_vector_exp ? vector_graph->process_event(ev)
                          : scalar_graph->process_event(ev);
  }

  void reset_decode_context() {
    if (!process_event(needle::graph::event::init{
            .activation_quant = true})) {
      fail_needle_setup("graph_init");
    }
    if (!process_event(needle::graph::event::prefill{
            std::span<const int32_t>{context_ids}, std::span<float>{logits}})) {
      fail_needle_setup("graph_context_prefill");
    }
    decoded_steps = 0u;
  }

  void set_swa_vector_exp(const bool enabled) {
    swa_vector_exp = enabled;
    reset_decode_context();
  }
};

template <class graph_type> struct benchmark_route_fixture {
  graph_type graph;
  std::vector<float> logits;
  std::vector<int32_t> context_ids;
  std::vector<int32_t> prompt_ids;
  uint32_t decoded_steps = 0u;

  explicit benchmark_route_fixture(const needle::contract &contract)
      : graph(contract), logits(contract.geo.vocab_size),
        context_ids(k_decode_context_tokens),
        prompt_ids(k_prefill_case_tokens) {
    for (size_t i = 0; i < context_ids.size(); ++i) {
      context_ids[i] =
          static_cast<int32_t>((1000003u * i + 7u) % contract.geo.vocab_size);
    }
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
      prompt_ids[i] =
          static_cast<int32_t>((1000033u * i + 13u) % contract.geo.vocab_size);
    }
  }

  void reset_decode_context() {
    if (!graph.process_event(
            needle::graph::event::init{.activation_quant = true})) {
      fail_needle_setup("graph_init");
    }
    if (!graph.process_event(needle::graph::event::prefill{
            std::span<const int32_t>{context_ids}, std::span<float>{logits}})) {
      fail_needle_setup("graph_context_prefill");
    }
    decoded_steps = 0u;
  }
};

template <class graph_type>
void append_graph_route_cases(std::vector<emel::bench::result> &results,
                              const emel::bench::config &cfg,
                              const needle::contract &contract,
                              const char *decode_name,
                              const char *prefill_name,
                              const char *backend_id,
                              const uint32_t thread_count,
                              const char *thread_contract,
                              const char *route_note) {
  {
    benchmark_route_fixture<graph_type> fixture{contract};
    emel::bench::config decode_cfg = cfg;
    decode_cfg.iterations = read_env_u64_or(k_decode_iters_env, 64u);
    decode_cfg.runs = cfg.runs;
    decode_cfg.warmup_iterations = 8u;
    decode_cfg.warmup_runs = 1u;

    const uint32_t step_budget =
        contract.geo.max_seq_len - k_decode_context_tokens - 2u;
    size_t token_cursor = 0u;
    auto reset_decode_run = [&]() {
      fixture.reset_decode_context();
      token_cursor = 0u;
    };
    auto decode_fn = [&]() {
      if (fixture.decoded_steps >= step_budget) {
        fixture.reset_decode_context();
      }
      const int32_t token = fixture.prompt_ids[token_cursor];
      token_cursor = (token_cursor + 1u) % fixture.prompt_ids.size();
      if (!fixture.graph.process_event(needle::graph::event::decode{
              token, std::span<float>{fixture.logits}})) {
        fail_needle_setup("graph_decode_step");
      }
      fixture.decoded_steps += 1u;
    };
    auto row = with_needle_metadata(
        emel::bench::measure_case_with_run_setup(
            decode_name, decode_cfg, reset_decode_run, decode_fn),
        "emel", backend_id, "cpp", 1u);
    row.thread_count = thread_count;
    row.thread_contract = thread_contract;
    row.note += route_note;
    results.push_back(std::move(row));
  }

  {
    benchmark_route_fixture<graph_type> fixture{contract};
    emel::bench::config prefill_cfg = cfg;
    prefill_cfg.iterations = read_env_u64_or(k_prefill_iters_env, 1u);
    prefill_cfg.runs = cfg.runs;
    prefill_cfg.warmup_iterations = 1u;
    prefill_cfg.warmup_runs = 1u;
    auto prefill_fn = [&]() {
      if (!fixture.graph.process_event(needle::graph::event::init{})) {
        fail_needle_setup("graph_prefill_init");
      }
      if (!fixture.graph.process_event(needle::graph::event::prefill{
              std::span<const int32_t>{fixture.prompt_ids},
              std::span<float>{fixture.logits}})) {
        fail_needle_setup("graph_prefill");
      }
    };
    auto row = with_needle_metadata(
        emel::bench::measure_case(prefill_name, prefill_cfg, prefill_fn),
        "emel", backend_id, "cpp", k_prefill_case_tokens);
    row.thread_count = thread_count;
    row.thread_contract = thread_contract;
    row.note += route_note;
    results.push_back(std::move(row));
  }
}

emel::bench::result with_needle_metadata(emel::bench::result out,
                                         const char *lane,
                                         const char *backend_id,
                                         const char *backend_language,
                                         const std::uint64_t output_tokens) {
  out.compare_group = out.name;
  out.lane = lane;
  out.backend_id = backend_id;
  out.backend_language = backend_language;
  out.thread_count = 1;
  out.thread_contract = "single_thread";
  out.comparison_mode = "measurement";
  out.model_id = k_model_id;
  out.fixture_id = k_model_relative_path;
  out.workload_id = k_workload_id;
  out.comparable = false;
  out.output_tokens = output_tokens;
  out.tokens_per_second =
      emel::bench::compute_tokens_per_second(output_tokens, out.ns_per_op);
  out.note = k_measurement_note;
  return out;
}

emel::bench::result with_internal_microbenchmark_metadata(
    emel::bench::result out, const char *backend_id) {
  out = with_needle_metadata(std::move(out), "emel", backend_id, "cpp", 1u);
  out.comparison_mode = "emel_internal_microbenchmark";
  out.note = k_internal_microbenchmark_note;
  return out;
}

emel::bench::result make_reference_row(const char *name,
                                       const std::uint64_t tokens,
                                       const double tokens_per_second) {
  emel::bench::result out;
  out.name = name;
  out.ns_per_op =
      static_cast<double>(tokens) * 1000000000.0 / tokens_per_second;
  out.ns_min_per_op = out.ns_per_op;
  out.ns_mean_per_op = out.ns_per_op;
  out.ns_max_per_op = out.ns_per_op;
  out.iterations = 1u;
  out.runs = 1u;
  return with_needle_metadata(std::move(out), "reference",
                              "libneedle_2_0_3_recorded", "recorded", tokens);
}


} // namespace

namespace emel::bench {

void append_emel_needle_graph_cases(std::vector<result> &results,
                                    const config &cfg) {
  graph_fixture fixture;
  const bool include_internal_microbenchmarks = cfg.mode != case_mode::compare;
  if (include_internal_microbenchmarks) {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__) && defined(__F16C__)
    append_hadamard_case(results, cfg, k_hadamard_scalar_case_name, false);
    append_hadamard_case(results, cfg, k_hadamard_avx2_case_name, true);
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    {
      config fwht_cfg = cfg;
      fwht_cfg.iterations = read_env_u64_or(k_fwht_iters_env, 100000u);
      fwht_cfg.runs = cfg.runs;
      fwht_cfg.warmup_iterations = 1000u;
      fwht_cfg.warmup_runs = 1u;
      alignas(32) std::array<float, 128u> values{};
      for (uint32_t i = 0u; i < values.size(); ++i)
        values[i] = std::sin(static_cast<float>(i + 1u) * 0.03125f);
      auto fwht_fn = [&]() {
        emel::kernel::cq::detail::fwht128_avx2(values.data());
        values[0] += 0.0000001f;
      };
      results.push_back(with_internal_microbenchmark_metadata(
          measure_case(k_fwht_case_name, fwht_cfg, fwht_fn),
          "emel_cq_avx2_fwht128"));
    }
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    append_swa_case(results, cfg, k_swa_scalar_exp_128_case_name, 128u,
                    swa_exp_route::scalar);
    append_swa_case(results, cfg, k_swa_vector_exp_128_case_name, 128u,
                    swa_exp_route::vector);
    append_swa_case(results, cfg, k_swa_scalar_exp_512_case_name, 512u,
                    swa_exp_route::scalar);
    append_swa_case(results, cfg, k_swa_vector_exp_512_case_name, 512u,
                    swa_exp_route::vector);
    append_swa_case(results, cfg, k_swa_scalar_exp_704_case_name, 704u,
                    swa_exp_route::scalar);
    append_swa_case(results, cfg, k_swa_vector_exp_704_case_name, 704u,
                    swa_exp_route::vector);
    if (instrument_graph()) {
      profile_swa_case(704u, swa_exp_route::scalar);
      profile_swa_case(704u, swa_exp_route::vector);
    }
#endif
  }

  append_graph_route_cases<needle::graph::serial_sm>(
      results, cfg, fixture.contract, k_serial_decode_case_name,
      k_serial_prefill_case_name, "emel_needle_graph_serial", 1u,
      "single_thread",
      " route=serial backend_id=emel_needle_graph_serial thread_count=1"
      " thread_contract=single_thread");
  append_graph_route_cases<needle::graph::parallel4_sm>(
      results, cfg, fixture.contract, k_parallel4_decode_case_name,
      k_parallel4_prefill_case_name, "emel_needle_graph_parallel4", 4u,
      "bounded_fork_join_3_workers_plus_owner",
      " route=parallel4 backend_id=emel_needle_graph_parallel4 thread_count=4"
      " thread_contract=bounded_fork_join_3_workers_plus_owner");

  if (instrument_graph() || instrument_cq()) {
    // Optional diagnostics retain the historical parallel4 route and do not
    // emit an additional benchmark row.
    config decode_cfg = cfg;
    decode_cfg.iterations = read_env_u64_or(k_decode_iters_env, 64u);
    decode_cfg.runs = cfg.runs;
    decode_cfg.warmup_iterations = 8u;
    decode_cfg.warmup_runs = 1u;
    const uint32_t step_budget =
        fixture.contract.geo.max_seq_len - k_decode_context_tokens - 2u;
    fixture.reset_decode_context();
    size_t token_cursor = 0u;
    auto decode_fn = [&]() {
      if (fixture.decoded_steps >= step_budget) {
        fixture.reset_decode_context();
      }
      const int32_t token = fixture.prompt_ids[token_cursor];
      token_cursor = (token_cursor + 1u) % fixture.prompt_ids.size();
      if (instrument_cq())
        fixture.process_event(needle::graph::event::configure_cq_timing{
            true, &benchmark_timestamp_now_ns});
      const auto decode_begin = std::chrono::steady_clock::now();
      if (!fixture.process_event(needle::graph::event::decode{
              token, std::span<float>{fixture.logits}})) {
        fail_needle_setup("graph_decode_step");
      }
      const auto decode_end = std::chrono::steady_clock::now();
      fixture.decode_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end -
                                                               decode_begin)
              .count());
      if (instrument_graph())
        fixture.process_event(
            needle::graph::event::capture_timing{fixture.graph_timing});
      else
        fixture.process_event(
            needle::graph::event::capture_cq_timing{fixture.cq_timing});
      fixture.decoded_steps += 1u;
    };
    if (instrument_graph()) {
      const auto print_graph_components = [&](const char *route) {
        const auto &t = fixture.graph_timing;
        const double tokens = static_cast<double>(t.steps);
        const auto print_phase = [&](const char *name, const uint64_t ns) {
          const double ns_per_token = tokens > 0.0 ? ns / tokens : 0.0;
          const double pct = t.total_nanoseconds > 0u
                                 ? static_cast<double>(ns) * 100.0 /
                                       static_cast<double>(t.total_nanoseconds)
                                 : 0.0;
          std::fprintf(stderr,
                       "# needle_graph_component: route=%s phase=%s ns=%llu "
                       "ns_per_token=%.3f pct_total=%.3f\n",
                       route, name, static_cast<unsigned long long>(ns),
                       ns_per_token, pct);
        };
        std::fprintf(stderr,
                     "# needle_graph_components: route=%s steps=%llu "
                     "measured_ns=%llu timed_ns=%llu reconciliation_pct=%.3f\n",
                     route, static_cast<unsigned long long>(t.steps),
                     static_cast<unsigned long long>(fixture.decode_ns),
                     static_cast<unsigned long long>(t.total_nanoseconds),
                     fixture.decode_ns > 0u
                         ? static_cast<double>(t.total_nanoseconds) * 100.0 /
                               static_cast<double>(fixture.decode_ns)
                         : 0.0);
        print_phase("cq", t.cq_nanoseconds);
        print_phase("graph_overhead", t.graph_overhead_nanoseconds);
        print_phase("engram", t.engram_nanoseconds);
        print_phase("norm", t.norm_nanoseconds);
        print_phase("mhc_pre", t.mhc_pre_nanoseconds);
        print_phase("mhc_post", t.mhc_post_nanoseconds);
        print_phase("attention_rope", t.attention_rope_nanoseconds);
        print_phase("attention_cache", t.attention_cache_nanoseconds);
        print_phase("attention_attend", t.attention_attend_nanoseconds);
        print_phase("attention_gate", t.attention_gate_nanoseconds);
        print_phase("hadamard", t.hadamard_nanoseconds);
        print_phase("lane_copy_mean", t.lane_copy_mean_nanoseconds);
        print_phase("sampling", t.sampling_nanoseconds);
      };
      const auto measure_graph_route = [&](const bool vector_exp) {
        fixture.set_swa_vector_exp(vector_exp);
        fixture.decode_ns = 0u;
        fixture.graph_timing = {};
        fixture.process_event(needle::graph::event::configure_timing{
            true, &benchmark_timestamp_now_ns});
        fixture.process_event(needle::graph::event::reset_timing{});
        token_cursor = 0u;
        const auto route_result = measure_case(k_parallel4_decode_case_name,
                                               decode_cfg, decode_fn);
        const double route_tokens_per_second =
            compute_tokens_per_second(1u, route_result.ns_per_op);
        const char *const route = vector_exp ? "vector_exp" : "scalar_exp";
        std::fprintf(stderr,
                     "# needle_graph_swa_route: route=%s ns_per_op=%.3f "
                     "tokens_per_second=%.3f\n",
                     route, route_result.ns_per_op, route_tokens_per_second);
        print_graph_components(route);
        fixture.process_event(
            needle::graph::event::configure_timing{false, nullptr});
      };
      for (uint32_t alternation = 0u; alternation < 5u; ++alternation) {
        measure_graph_route(false);
        measure_graph_route(true);
      }
    }
    if (instrument_cq()) {
      const auto &t = fixture.cq_timing;
      const std::uint64_t cq_ns =
          t.quantize_nanoseconds + t.fwht_nanoseconds +
          t.dot_full_nanoseconds + t.dot_batch_nanoseconds +
          t.dot_rows_nanoseconds + t.dequant_nanoseconds;
      const double pct = static_cast<double>(cq_ns) * 100.0 /
                         static_cast<double>(fixture.decode_ns);
      std::fprintf(
          stderr,
          "# needle_graph_cq: decode_ns=%llu quant_ns=%llu fwht_ns=%llu "
          "dot_full_ns=%llu dot_batch_ns=%llu dot_rows_ns=%llu "
          "dequant_ns=%llu cq_ns=%llu cq_pct=%.3f non_cq_ns=%llu\n",
          static_cast<unsigned long long>(fixture.decode_ns),
          static_cast<unsigned long long>(t.quantize_nanoseconds),
          static_cast<unsigned long long>(t.fwht_nanoseconds),
          static_cast<unsigned long long>(t.dot_full_nanoseconds),
          static_cast<unsigned long long>(t.dot_batch_nanoseconds),
          static_cast<unsigned long long>(t.dot_rows_nanoseconds),
          static_cast<unsigned long long>(t.dequant_nanoseconds),
          static_cast<unsigned long long>(cq_ns), pct,
          static_cast<unsigned long long>(fixture.decode_ns - cq_ns));
    }
  }
}

void append_reference_needle_graph_cases(std::vector<result> &results,
                                         const config &) {
  results.push_back(make_reference_row(k_serial_decode_case_name, 1u,
                                       k_libneedle_decode_tokens_per_second));
  results.push_back(make_reference_row(k_parallel4_decode_case_name, 1u,
                                       k_libneedle_decode_tokens_per_second));
  results.push_back(make_reference_row(k_serial_prefill_case_name,
                                       k_prefill_case_tokens,
                                       k_libneedle_prefill_tokens_per_second));
  results.push_back(make_reference_row(k_parallel4_prefill_case_name,
                                       k_prefill_case_tokens,
                                       k_libneedle_prefill_tokens_per_second));
}

} // namespace emel::bench
