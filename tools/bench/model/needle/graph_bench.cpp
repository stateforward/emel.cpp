// Needle graph decode/prefill single-core benchmark on the pinned
// tests/models/route-w4-qat.cact fixture. The EMEL lane drives the native
// graph machine ONLY through its public events (init/prefill/decode) via the
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

constexpr char k_decode_case_name[] = "needle/graph/decode_steady_route_w4_qat";
constexpr char k_prefill_case_name[] = "needle/graph/prefill_512_route_w4_qat";
constexpr char k_fwht_case_name[] = "needle/cq/fwht128_avx2";
constexpr char k_fwht_iters_env[] = "EMEL_BENCH_NEEDLE_FWHT_ITERS";
constexpr char k_model_env[] = "EMEL_BENCH_NEEDLE_MODEL";
constexpr char k_model_relative_path[] = "tests/models/route-w4-qat.cact";
constexpr char k_model_id[] = "route_w4_qat_cact";
constexpr char k_workload_id[] = "needle_graph_single_core_v1";
constexpr char k_decode_iters_env[] = "EMEL_BENCH_NEEDLE_GRAPH_DECODE_ITERS";
constexpr char k_prefill_iters_env[] = "EMEL_BENCH_NEEDLE_GRAPH_PREFILL_ITERS";
constexpr char k_instrument_cq_env[] = "EMEL_BENCH_NEEDLE_GRAPH_INSTRUMENT_CQ";

// Steady-state decode is measured after a realistic ~100-token prefill; the
// prefill case runs a 512-token prompt (max_seq_len 2048 bounded).
constexpr uint32_t k_decode_context_tokens = 100u;
constexpr uint32_t k_prefill_case_tokens = 512u;

// Documented libneedle reference lane constants (see file header note).
constexpr double k_libneedle_decode_tokens_per_second = 145.0;
constexpr double k_libneedle_prefill_tokens_per_second = 180.0;

constexpr char k_measurement_note[] =
    "proof_status=measurement_only "
    "reference=libneedle_2.0.3_recorded "
    "source=user_supplied_single_core_same_host_class "
    "target_decode_tps=435";

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

void on_loader_probe_done(const cact_loader::events::probe_done &) {}
void on_loader_probe_error(const cact_loader::events::probe_error &) {}
void on_loader_bind_done(const cact_loader::events::bind_done &) {}
void on_loader_bind_error(const cact_loader::events::bind_error &) {}
void on_loader_parse_done(const cact_loader::events::parse_done &) {}
void on_loader_parse_error(const cact_loader::events::parse_error &) {}
void on_needle_bind_done(const needle::events::bind_done &) {}
void on_needle_bind_error(const needle::events::bind_error &) {}

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
  std::unique_ptr<needle::graph::sm> graph;
  std::vector<float> logits;
  std::vector<int32_t> context_ids;
  std::vector<int32_t> prompt_ids;
  uint32_t decoded_steps = 0u;
  std::uint64_t decode_ns = 0u;
  emel::kernel::cq::event::timing_breakdown cq_timing = {};

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

    graph = std::make_unique<needle::graph::sm>(contract);
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

  void reset_decode_context() {
    if (!graph->process_event(needle::graph::event::init{})) {
      fail_needle_setup("graph_init");
    }
    if (!graph->process_event(needle::graph::event::prefill{
            std::span<const int32_t>{context_ids}, std::span<float>{logits}})) {
      fail_needle_setup("graph_context_prefill");
    }
    decoded_steps = 0u;
  }
};

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
    results.push_back(with_needle_metadata(
        measure_case(k_fwht_case_name, fwht_cfg, fwht_fn), "emel",
        "emel_cq_avx2_fwht128", "cpp", 1u));
  }
#endif

  {
    // Steady-state decode: one greedy-shaped decode step per op at ~100-token
    // context depth. The fixture re-arms (init + context prefill) whenever the
    // position budget nears max_seq_len so arbitrary iteration counts stay
    // inside the graph's step-validity guard.
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
        fixture.graph->process_event(needle::graph::event::configure_cq_timing{
            true, &benchmark_timestamp_now_ns});
      const auto decode_begin = std::chrono::steady_clock::now();
      if (!fixture.graph->process_event(needle::graph::event::decode{
              token, std::span<float>{fixture.logits}})) {
        fail_needle_setup("graph_decode_step");
      }
      const auto decode_end = std::chrono::steady_clock::now();
      if (instrument_cq()) {
        fixture.decode_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end -
                                                                 decode_begin)
                .count());
        fixture.graph->process_event(
            needle::graph::event::capture_cq_timing{fixture.cq_timing});
      }
      fixture.decoded_steps += 1u;
    };
    auto decode_result =
        measure_case(k_decode_case_name, decode_cfg, decode_fn);
    results.push_back(with_needle_metadata(std::move(decode_result), "emel",
                                           "emel_needle_graph", "cpp", 1u));
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

  {
    // Prefill: one 512-token prompt per op; init is part of the op so every
    // iteration prefills from an empty cache (init cost is negligible next to
    // 512 layer-stack steps).
    config prefill_cfg = cfg;
    prefill_cfg.iterations = read_env_u64_or(k_prefill_iters_env, 1u);
    prefill_cfg.runs = cfg.runs;
    prefill_cfg.warmup_iterations = 1u;
    prefill_cfg.warmup_runs = 1u;

    auto prefill_fn = [&]() {
      if (!fixture.graph->process_event(needle::graph::event::init{})) {
        fail_needle_setup("graph_prefill_init");
      }
      if (!fixture.graph->process_event(needle::graph::event::prefill{
              std::span<const int32_t>{fixture.prompt_ids},
              std::span<float>{fixture.logits}})) {
        fail_needle_setup("graph_prefill");
      }
    };
    results.push_back(with_needle_metadata(
        measure_case(k_prefill_case_name, prefill_cfg, prefill_fn), "emel",
        "emel_needle_graph", "cpp", k_prefill_case_tokens));
  }
}

void append_reference_needle_graph_cases(std::vector<result> &results,
                                         const config &) {
  results.push_back(make_reference_row(k_decode_case_name, 1u,
                                       k_libneedle_decode_tokens_per_second));
  results.push_back(make_reference_row(k_prefill_case_name,
                                       k_prefill_case_tokens,
                                       k_libneedle_prefill_tokens_per_second));
}

} // namespace emel::bench
