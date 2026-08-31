#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/cact/loader/sm.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/model/needle/sm.hpp"

namespace {

// Committed W4/f32 parity fixture (route-w4-qat.logits.json): 3 cases, each
// 1 prefill + 2 greedy decode steps. W4A8 deployment parity is covered by the
// separate route-w4-qat-a8 fixture test below.
constexpr uint32_t k_vocab = 8192u;
constexpr uint32_t k_steps = 3u;

struct parity_case {
  std::vector<int32_t> prompt_ids;
  std::array<int32_t, k_steps> greedy;
  const char *file;
};

const parity_case k_cases[3] = {
    {{2, 1544, 1663, 2328}, {8097, 341, 359}, "route-w4-qat.logits.case0.bin"},
    {{2, 5722, 625, 5019}, {8063, 24, 7}, "route-w4-qat.logits.case1.bin"},
    {{2, 7551}, {8097, 2730, 8097}, "route-w4-qat.logits.case2.bin"},
};

const parity_case k_a8_cases[3] = {
    {{2, 1544, 1663, 2328},
     {8097, 341, 359},
     "route-w4-qat-a8.logits.case0.bin"},
    {{2, 5722, 625, 5019}, {8063, 24, 7}, "route-w4-qat-a8.logits.case1.bin"},
    {{2, 7551}, {8097, 2730, 8097}, "route-w4-qat-a8.logits.case2.bin"},
};

void on_loader_probe_done(const emel::cact::loader::events::probe_done &) {}
void on_loader_probe_error(const emel::cact::loader::events::probe_error &) {}
void on_loader_bind_done(const emel::cact::loader::events::bind_done &) {}
void on_loader_bind_error(const emel::cact::loader::events::bind_error &) {}
void on_loader_parse_done(const emel::cact::loader::events::parse_done &) {}
void on_loader_parse_error(const emel::cact::loader::events::parse_error &) {}
void on_needle_bind_done(const emel::model::needle::events::bind_done &) {}
void on_needle_bind_error(const emel::model::needle::events::bind_error &) {}

const emel::cact::loader::event::probe_done_fn k_probe_done =
    emel::cact::loader::event::probe_done_fn::from<&on_loader_probe_done>();
const emel::cact::loader::event::probe_error_fn k_probe_error =
    emel::cact::loader::event::probe_error_fn::from<&on_loader_probe_error>();
const emel::cact::loader::event::bind_done_fn k_bind_done =
    emel::cact::loader::event::bind_done_fn::from<&on_loader_bind_done>();
const emel::cact::loader::event::bind_error_fn k_bind_error =
    emel::cact::loader::event::bind_error_fn::from<&on_loader_bind_error>();
const emel::cact::loader::event::parse_done_fn k_parse_done =
    emel::cact::loader::event::parse_done_fn::from<&on_loader_parse_done>();
const emel::cact::loader::event::parse_error_fn k_parse_error =
    emel::cact::loader::event::parse_error_fn::from<&on_loader_parse_error>();
const emel::model::needle::event::bind_done_fn k_needle_done =
    emel::model::needle::event::bind_done_fn::from<&on_needle_bind_done>();
const emel::model::needle::event::bind_error_fn k_needle_error =
    emel::model::needle::event::bind_error_fn::from<&on_needle_bind_error>();

std::vector<uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  REQUIRE(size > 0);
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(input.good());
  return bytes;
}

std::vector<float> read_reference_logits(const char *name) {
  const auto path =
      std::filesystem::path{EMEL_TEST_REPO_ROOT} / "tests/fixtures/cact" / name;
  const auto bytes = read_file_bytes(path);
  REQUIRE(bytes.size() == static_cast<size_t>(k_steps) * k_vocab * 4u);
  std::vector<float> values(static_cast<size_t>(k_steps) * k_vocab);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

uint32_t argmax(const std::span<const float> logits) {
  uint32_t best = 0u;
  for (uint32_t i = 1u; i < logits.size(); ++i)
    best = logits[i] > logits[best] ? i : best;
  return best;
}

struct step_error {
  double max_abs = 0.0;
  double rel = 0.0;
};

step_error compare_step(const std::span<const float> native,
                        const std::span<const float> reference) {
  step_error result{};
  double max_ref = 0.0;
  for (uint32_t i = 0u; i < k_vocab; ++i) {
    const double diff = std::abs(static_cast<double>(native[i]) -
                                 static_cast<double>(reference[i]));
    result.max_abs = diff > result.max_abs ? diff : result.max_abs;
    const double magnitude = std::abs(static_cast<double>(reference[i]));
    max_ref = magnitude > max_ref ? magnitude : max_ref;
  }
  result.rel = result.max_abs / max_ref;
  return result;
}

} // namespace

uint64_t g_timing_clock = 0u;

uint64_t fake_timestamp_now() noexcept {
  g_timing_clock += 10u;
  return g_timing_clock;
}

TEST_CASE("needle graph matches the committed JAX logits fixture on all "
          "cases") {
  // Load the pinned .cact through the maintained loader chain.
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);

  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));

  // Bind the named contract.
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));
  REQUIRE(contract.geo.vocab_size == k_vocab);

  // The graph machine allocates all runtime storage at construction.
  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  double worst_abs = 0.0;
  double worst_rel = 0.0;

  for (const auto &parity : k_cases) {
    // Re-init between cases: clears KV caches, lanes, and history.
    REQUIRE(graph.process_event(
        emel::model::needle::graph::event::init{.activation_quant = false}));

    const auto reference = read_reference_logits(parity.file);
    std::array<int32_t, k_steps> greedy = {};

    REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
        std::span<const int32_t>{parity.prompt_ids},
        std::span<float>{logits}}));
    for (uint32_t step = 0u; step < k_steps; ++step) {
      const std::span<const float> reference_step{
          reference.data() + static_cast<size_t>(step) * k_vocab, k_vocab};
      const step_error err = compare_step(logits, reference_step);
      worst_abs = err.max_abs > worst_abs ? err.max_abs : worst_abs;
      worst_rel = err.rel > worst_rel ? err.rel : worst_rel;
      greedy[step] = static_cast<int32_t>(
          argmax(std::span<const float>{logits.data(), k_vocab}));
      MESSAGE("case ", std::string_view{parity.file}, " step ", step,
              ": max_abs=", err.max_abs, " rel=", err.rel,
              " argmax=", greedy[step]);
      // Target tolerance: rel <= 1e-3 against the max reference magnitude.
      CHECK(err.rel <= 1e-3);
      CHECK(greedy[step] == parity.greedy[step]);
      if (step + 1u < k_steps) {
        REQUIRE(graph.process_event(emel::model::needle::graph::event::decode{
            greedy[step], std::span<float>{logits}}));
      }
    }
  }
  MESSAGE("worst-case parity across 3 cases x 3 steps: max_abs=", worst_abs,
          " rel=", worst_rel);
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  uint64_t gqa2_calls = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_swa_diagnostics{gqa2_calls}));
  CHECK(gqa2_calls > 0u);
#endif
}
TEST_CASE("needle graph W4A8 deployment route matches authoritative JAX "
          "fixture") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  double worst_abs = 0.0;
  double worst_rel = 0.0;
  for (const auto &parity : k_a8_cases) {
    REQUIRE(graph.process_event(emel::model::needle::graph::event::init{}));
    const auto reference = read_reference_logits(parity.file);
    std::array<int32_t, k_steps> greedy = {};
    REQUIRE(graph.process_event(emel::model::needle::graph::event::prefill{
        std::span<const int32_t>{parity.prompt_ids},
        std::span<float>{logits}}));
    for (uint32_t step = 0u; step < k_steps; ++step) {
      const std::span<const float> reference_step{
          reference.data() + static_cast<size_t>(step) * k_vocab, k_vocab};
      const step_error err = compare_step(logits, reference_step);
      worst_abs = err.max_abs > worst_abs ? err.max_abs : worst_abs;
      worst_rel = err.rel > worst_rel ? err.rel : worst_rel;
      greedy[step] = static_cast<int32_t>(
          argmax(std::span<const float>{logits.data(), k_vocab}));
      MESSAGE("A8 case ", std::string_view{parity.file}, " step ", step,
              ": max_abs=", err.max_abs, " rel=", err.rel,
              " argmax=", greedy[step]);
      // The native graph keeps exact CQ operands and greedy identity. Its
      // scalar stage ordering differs from XLA, so use the generated fixture's
      // observed 1.5e-2 relative envelope rather than the legacy f32 1e-3.
      CHECK(err.rel <= 1.5e-2);
      CHECK(greedy[step] == parity.greedy[step]);
      if (step + 1u < k_steps) {
        REQUIRE(graph.process_event(emel::model::needle::graph::event::decode{
            greedy[step], std::span<float>{logits}}));
      }
    }
  }
  MESSAGE("worst-case W4A8 parity across 3 cases x 3 steps: max_abs=",
          worst_abs, " rel=", worst_rel);

  uint64_t quantize_calls = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_a8_diagnostics{
          quantize_calls}));
  CHECK(quantize_calls > 0u);
}

TEST_CASE("needle graph prepares CQ4 storage once and selects prepared route") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  uint64_t prepare_calls = 0u;
  uint64_t prepared_calls = 0u;
  size_t prepared_index_bytes = 0u;
  size_t prepared_input32_bytes = 0u;
  size_t prepared_norm_bytes = 0u;
  size_t prepared_group32_norm_bytes = 0u;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepare_calls > 0u);
  CHECK(prepared_calls == 0u);
  CHECK(prepared_index_bytes > 0u);
  CHECK(prepared_input32_bytes == prepared_index_bytes);
  CHECK(prepared_norm_bytes > 0u);
  CHECK(prepared_group32_norm_bytes > 0u);

  std::vector<float> logits(k_vocab);
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  const uint64_t prepared_after_decode = prepared_calls;
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepared_calls > prepared_after_decode);
  const uint64_t preparation_count = prepare_calls;

  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::decode{2, std::span<float>{logits}}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_cq_diagnostics{
          prepare_calls, prepared_calls, prepared_index_bytes,
          prepared_input32_bytes, prepared_norm_bytes,
          prepared_group32_norm_bytes}));
  CHECK(prepare_calls == preparation_count);
}

TEST_CASE("needle graph component timing is explicit, resettable, and reconciled") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  std::vector<float> logits(k_vocab);
  REQUIRE(graph.process_event(emel::model::needle::graph::event::init{}));
  g_timing_clock = 0u;
  REQUIRE(graph.process_event(emel::model::needle::graph::event::configure_timing{
      true, &fake_timestamp_now}));
  REQUIRE(graph.process_event(emel::model::needle::graph::event::decode{
      2, std::span<float>{logits}}));
  emel::model::needle::graph::event::timing_breakdown timing{};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{timing}));
  CHECK(timing.steps == 1u);
  CHECK(timing.total_nanoseconds > 0u);
  const uint64_t split =
      timing.cq_nanoseconds + timing.graph_overhead_nanoseconds +
      timing.engram_nanoseconds + timing.norm_nanoseconds +
      timing.mhc_pre_nanoseconds + timing.mhc_post_nanoseconds +
      timing.attention_rope_nanoseconds + timing.attention_cache_nanoseconds +
      timing.attention_attend_nanoseconds + timing.attention_gate_nanoseconds +
      timing.hadamard_nanoseconds + timing.lane_copy_mean_nanoseconds +
      timing.sampling_nanoseconds;
  CHECK(split == timing.total_nanoseconds);
  REQUIRE(graph.process_event(emel::model::needle::graph::event::reset_timing{}));
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::capture_timing{timing}));
  CHECK(timing.steps == 0u);
  CHECK(timing.total_nanoseconds == 0u);
  REQUIRE(graph.process_event(emel::model::needle::graph::event::configure_timing{
      false, nullptr}));
}

TEST_CASE("needle graph AVX2 route requires every CQ tensor group to be 128") {
  emel::model::needle::contract contract{};
  contract.layer_count = 1u;
  contract.engram_site_count = 1u;
  contract.geo.d_model = 512u;
  contract.geo.hada_n = 512u;
  contract.embedding.group = 128u;
  contract.mhc.phi_pre.group = 128u;
  contract.mhc.phi_post.group = 128u;
  contract.mhc.phi_res.group = 128u;
  auto &layer = contract.layers[0];
  layer.q_proj.group = 128u;
  layer.k_proj.group = 128u;
  layer.v_proj.group = 128u;
  layer.gate_proj.group = 128u;
  layer.out_proj.group = 128u;
  auto &site = contract.engram_sites[0];
  site.tables.group = 128u;
  site.key_proj.group = 128u;
  site.value_proj.group = 128u;

  emel::model::needle::graph::action::context ctx{contract};
  emel::model::needle::graph::event::step_ctx step{};
  const emel::model::needle::graph::event::step_run run{step};
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__) && defined(__F16C__)
  CHECK(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK_FALSE(
      emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
#else
  CHECK_FALSE(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
#endif

  layer.out_proj.group = 64u;
  CHECK_FALSE(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));

  layer.out_proj.group = 128u;
  contract.geo.d_model = 256u;
  CHECK_FALSE(emel::model::needle::graph::guard::guard_route_avx2{}(run, ctx));
  CHECK(emel::model::needle::graph::guard::guard_route_scalar{}(run, ctx));
}

TEST_CASE("needle graph rejects an out-of-vocab step token") {
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);

  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));

  emel::model::needle::graph::sm graph{contract};
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  std::vector<float> logits(k_vocab);
  CHECK_FALSE(graph.process_event(emel::model::needle::graph::event::decode{
      static_cast<int32_t>(k_vocab), std::span<float>{logits}}));
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_errored>));

  // Re-init recovers the machine.
  REQUIRE(graph.process_event(
      emel::model::needle::graph::event::init{.activation_quant = false}));
  CHECK(graph.is(
      stateforward::sml::state<emel::model::needle::graph::state_ready>));
}

TEST_CASE("needle graph serial and parallel4 routes are exact deterministic peers") {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__) && defined(__F16C__)
  const auto model_path = std::filesystem::path{EMEL_TEST_REPO_ROOT} /
                          "tests/models/route-w4-qat.cact";
  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  REQUIRE(loader.process_event(
      emel::cact::loader::event::probe{std::span<const uint8_t>{file_bytes},
                                       geometry, k_probe_done, k_probe_error}));
  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  REQUIRE(loader.process_event(emel::cact::loader::event::bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors}, k_bind_done,
      k_bind_error}));
  REQUIRE(loader.process_event(emel::cact::loader::event::parse{
      std::span<const uint8_t>{file_bytes}, k_parse_done, k_parse_error}));
  emel::model::needle::sm binder{};
  emel::model::needle::contract contract = {};
  REQUIRE(binder.process_event(emel::model::needle::event::bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_needle_done, k_needle_error}));
  emel::model::needle::graph::serial_sm serial{contract};
  emel::model::needle::graph::parallel4_sm parallel4{contract};
  for (const bool activation_quant : {false, true}) {
    REQUIRE(serial.process_event(
        emel::model::needle::graph::event::init{activation_quant}));
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::init{activation_quant}));
    std::vector<float> serial_logits(k_vocab);
    std::vector<float> parallel4_logits(k_vocab);
    const std::array<int32_t, 4u> prompt = {2, 1544, 1663, 2328};
    REQUIRE(serial.process_event(emel::model::needle::graph::event::prefill{
        prompt, serial_logits}));
    REQUIRE(parallel4.process_event(emel::model::needle::graph::event::prefill{
        prompt, parallel4_logits}));
    CHECK(parallel4_logits == serial_logits);
    const int32_t next = static_cast<int32_t>(argmax(serial_logits));
    REQUIRE(serial.process_event(emel::model::needle::graph::event::decode{
        next, serial_logits}));
    REQUIRE(parallel4.process_event(emel::model::needle::graph::event::decode{
        next, parallel4_logits}));
    CHECK(parallel4_logits == serial_logits);
    std::array<uint64_t, 3u> calls{};
    uint64_t submitted = 0u;
    uint64_t joined = 0u;
    uint64_t live = 1u;
    REQUIRE(parallel4.process_event(
        emel::model::needle::graph::event::capture_projection_diagnostics{
            calls, submitted, joined, live}));
    CHECK(calls[0] > 0u);
    CHECK(calls[1] > 0u);
    CHECK(calls[2] > 0u);
    CHECK(submitted == joined);
    CHECK(live == 0u);
  }
#endif
}
