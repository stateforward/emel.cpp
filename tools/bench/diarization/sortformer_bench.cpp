#include "bench_cases.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "diarization/sortformer_fixture.hpp"
#include "emel/diarization/sortformer/request/sm.hpp"
#include "emel/diarization/sortformer/pipeline/any.hpp"

namespace emel::bench {

namespace {

namespace fixture = emel::bench::diarization::sortformer_fixture;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace request = emel::diarization::sortformer::request;

constexpr const char *k_proof_status = "maintained_loader_real_audio";
constexpr const char *k_emel_backend_id = "emel.diarization.sortformer";
constexpr const char *k_reference_backend_id = "recorded.diarization.baseline";
constexpr const char *k_onnx_feature_input_env =
    "EMEL_DIARIZATION_ONNX_FEATURE_INPUT";
constexpr const char *k_onnx_encoder_probe_env =
    "EMEL_DIARIZATION_ONNX_ENCODER_PROBE_OUTPUT";
constexpr const char *k_hidden_probe_env =
    "EMEL_DIARIZATION_HIDDEN_PROBE_OUTPUT";
constexpr const char *k_probability_probe_env =
    "EMEL_DIARIZATION_PROBABILITY_PROBE_OUTPUT";
constexpr const char *k_stage_profile_env = "EMEL_DIARIZATION_STAGE_PROFILE";
constexpr const char *k_diarization_iters_env = "EMEL_BENCH_DIARIZATION_ITERS";
constexpr const char *k_diarization_runs_env = "EMEL_BENCH_DIARIZATION_RUNS";
volatile std::uint64_t g_checksum_sink = 0;
volatile bool g_stage_ok_sink = false;

[[noreturn]] void fail_sortformer_setup(const char *step) {
  std::fprintf(stderr,
               "error: sortformer diarization bench setup failed at %s "
               "(model=%s audio=%s baseline=%s)\n",
               step, fixture::k_model_rel_path.data(),
               fixture::k_audio_rel_path.data(),
               fixture::k_baseline_rel_path.data());
  std::exit(1);
}

void require_model_fixture(fixture::model_fixture &model) {
  if (!fixture::prepare(model)) {
    fail_sortformer_setup("prepare_model_fixture");
  }
}

void require_pcm_fixture(fixture::pcm_fixture &pcm) {
  if (!fixture::prepare(pcm)) {
    fail_sortformer_setup("prepare_pcm_fixture");
  }
}

std::uint64_t read_env_u64_or(const char *name, const std::uint64_t fallback) noexcept {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0') || parsed == 0) {
    return fallback;
  }
  return static_cast<std::uint64_t>(parsed);
}

std::size_t read_env_size_or(const char *name, const std::size_t fallback) noexcept {
  return static_cast<std::size_t>(
      read_env_u64_or(name, static_cast<std::uint64_t>(fallback)));
}

void require_baseline(fixture::expected_output_baseline &baseline) {
  if (!fixture::prepare(baseline)) {
    fail_sortformer_setup("prepare_expected_output_baseline");
  }
}

void write_float_probe_file(const char *output_path,
                            std::span<const float> values) {
  if (output_path == nullptr || output_path[0] == '\0') {
    return;
  }

  const std::filesystem::path path(output_path);
  std::error_code ec = {};
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      fail_sortformer_setup("onnx_probe_output_dir");
    }
  }
  std::ofstream output(path, std::ios::binary);
  if (!output.good()) {
    fail_sortformer_setup("onnx_probe_output_open");
  }
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output.good()) {
    fail_sortformer_setup("onnx_probe_output_write");
  }
}

bool env_value_requested(const char *value) noexcept {
  return value != nullptr && value[0] != '\0';
}

void maybe_write_onnx_probe_inputs(const fixture::model_fixture &model,
                                   const fixture::pcm_fixture &pcm) {
  const char *feature_path = std::getenv(k_onnx_feature_input_env);
  const char *encoder_path = std::getenv(k_onnx_encoder_probe_env);
  const char *hidden_path = std::getenv(k_hidden_probe_env);
  if (!env_value_requested(feature_path) &&
      !env_value_requested(encoder_path) &&
      !env_value_requested(hidden_path)) {
    return;
  }
  if (env_value_requested(encoder_path) || env_value_requested(hidden_path)) {
    fail_sortformer_setup("stage_probe_requires_public_stage_actor");
  }

  std::vector<float> features(
      static_cast<size_t>(pipeline::k_required_feature_count));
  int32_t feature_frame_count = 0;
  int32_t feature_bin_count = 0;
  emel::error::type feature_err =
      emel::error::cast(request::error::none);
  request::sm feature_machine{};
  request::event::prepare feature_ev{
      model.contract,
      pcm.pcm,
      pcm.sample_rate,
      pipeline::k_channel_count,
      features,
      feature_frame_count,
      feature_bin_count,
  };
  feature_ev.error_out = &feature_err;
  if (!feature_machine.process_event(feature_ev) ||
      feature_err != emel::error::cast(request::error::none) ||
      feature_frame_count != pipeline::k_feature_frame_count ||
      feature_bin_count != pipeline::k_feature_bin_count) {
    fail_sortformer_setup("onnx_feature_compute");
  }
  write_float_probe_file(feature_path, features);
}

config pipeline_benchmark_config(const config &cfg) noexcept {
  config out = cfg;
  out.iterations = read_env_u64_or(k_diarization_iters_env, 1u);
  out.runs = read_env_size_or(k_diarization_runs_env, std::min<std::size_t>(cfg.runs, 3u));
  return out;
}

result with_sortformer_metadata(result out, const char *lane,
                                const char *backend_id,
                                const char *backend_language,
                                const std::uint64_t checksum,
                                const std::uint64_t output_dim,
                                const char *proof_status) {
  out.lane = lane;
  out.backend_id = backend_id;
  out.backend_language = backend_language;
  out.compare_group = fixture::k_case_name;
  out.comparison_mode = "parity";
  out.model_id = fixture::k_model_id;
  out.fixture_id = fixture::k_fixture_id;
  out.workload_id = "diarization_sortformer_pipeline_v1";
  out.comparable = true;
  out.output_dim = output_dim;
  out.output_checksum = checksum;
  out.note =
      std::string{fixture::k_profile_id} + " proof_status=" + proof_status;
  return out;
}

std::string format_segments(
    std::span<const pipeline::segment_record> segments,
    const int32_t segment_count) {
  std::string out = {};
  for (int32_t i = 0; i < segment_count; ++i) {
    const auto &segment = segments[static_cast<size_t>(i)];
    out += "speaker=" + std::to_string(segment.speaker);
    out += " start_frame=" + std::to_string(segment.start_frame);
    out += " end_frame=" + std::to_string(segment.end_frame);
    out += '\n';
  }
  return out;
}

template <class fn_type> double measure_once_ns(fn_type &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count());
}

void set_single_sample_stats(result &out) noexcept {
  out.ns_min_per_op = out.ns_per_op;
  out.ns_mean_per_op = out.ns_per_op;
  out.ns_max_per_op = out.ns_per_op;
}

result make_profile_reference_result(
    const char *lane, const fixture::expected_output_baseline &baseline) {
  result out;
  out.name = fixture::k_profile_case_name;
  out.compare_group = fixture::k_profile_case_name;
  out.lane = lane;
  out.backend_id = std::string_view{lane} == "emel" ? k_emel_backend_id
                                                    : k_reference_backend_id;
  out.backend_language = std::string_view{lane} == "emel" ? "cpp" : "recorded";
  out.comparison_mode = "measurement";
  out.model_id = fixture::k_model_id;
  out.fixture_id = fixture::k_fixture_id;
  out.workload_id = "diarization_sortformer_pipeline_profile_v1";
  out.comparable = false;
  out.iterations = 1;
  out.runs = 1;
  set_single_sample_stats(out);
  out.output_dim = static_cast<std::uint64_t>(baseline.segment_count);
  out.output_checksum = baseline.output_checksum;
  out.note =
      "profile=orchestrated_pipeline source=recorded_maintained_baseline "
      "proof_status=measurement_only";
  return out;
}

result
make_profile_emel_result(const fixture::model_fixture &model,
                         const fixture::pcm_fixture &pcm,
                         const fixture::expected_output_baseline &baseline,
                         const double prepare_ns) {
  fixture::run_result pipeline_result{};
  auto machine = std::make_unique<pipeline::sm>();

  const double pipeline_ns = measure_once_ns([&]() {
    g_stage_ok_sink = fixture::run_pipeline(*machine, model.contract, pcm.pcm,
                                            pcm.sample_rate, pipeline_result);
    g_checksum_sink = fixture::compute_checksum(pipeline_result.segments,
                                                pipeline_result.segment_count);
  });

  if (!g_stage_ok_sink ||
      pipeline_result.err != emel::error::cast(pipeline::error::none) ||
      pipeline_result.segment_count <= 0) {
    fail_sortformer_setup("profile_pipeline_run");
  }

  result out = make_profile_reference_result("emel", baseline);
  out.backend_id = k_emel_backend_id;
  out.backend_language = "cpp";
  out.comparison_mode = "measurement";
  out.ns_per_op = pipeline_ns;
  set_single_sample_stats(out);
  out.prepare_ns_per_op = prepare_ns;
  out.encode_ns_per_op = pipeline_ns;
  out.publish_ns_per_op = 0.0;
  out.output_dim = static_cast<std::uint64_t>(pipeline_result.segment_count);
  out.output_checksum = fixture::compute_checksum(
      pipeline_result.segments, pipeline_result.segment_count);
  out.output_text =
      format_segments(pipeline_result.segments, pipeline_result.segment_count);
  out.note += " end_to_end_ns=" +
              std::to_string(static_cast<std::uint64_t>(out.ns_per_op));
  out.note +=
      " prepare_ns=" + std::to_string(static_cast<std::uint64_t>(prepare_ns));
  out.note += " runtime_error=" +
              std::to_string(static_cast<std::uint32_t>(pipeline_result.err));
  out.note += " load_strategy=";
  out.note +=
      emel::tools::model_load_io_strategy_name(model.load.used_io_strategy);
  return out;
}

result
make_stage_profile_result(const fixture::model_fixture &model,
                          const fixture::pcm_fixture &pcm,
                          const fixture::expected_output_baseline &baseline) {
  fixture::run_result pipeline_result{};
  auto machine = std::make_unique<pipeline::sm>();

  const double pipeline_ns = measure_once_ns([&]() {
    g_stage_ok_sink = fixture::run_pipeline(*machine, model.contract, pcm.pcm,
                                            pcm.sample_rate, pipeline_result);
    g_checksum_sink = fixture::compute_checksum(pipeline_result.segments,
                                                pipeline_result.segment_count);
  });
  if (!g_stage_ok_sink ||
      pipeline_result.err != emel::error::cast(pipeline::error::none) ||
      pipeline_result.segment_count <= 0) {
    fail_sortformer_setup("stage_profile_pipeline_run");
  }

  result out = make_profile_reference_result("emel", baseline);
  out.name = "diarization/sortformer/"
             "stage_profile_ami_en2002b_mix_headset_137.00_152.04_16khz_mono";
  out.compare_group = out.name;
  out.backend_id = k_emel_backend_id;
  out.backend_language = "cpp";
  out.comparison_mode = "measurement";
  out.comparable = false;
  out.workload_id = "diarization_sortformer_public_pipeline_profile_v1";
  out.ns_per_op = pipeline_ns;
  set_single_sample_stats(out);
  out.prepare_ns_per_op = 0.0;
  out.encode_ns_per_op = pipeline_ns;
  out.publish_ns_per_op = 0.0;
  out.output_dim = static_cast<std::uint64_t>(pipeline_result.segment_count);
  out.output_checksum = fixture::compute_checksum(
      pipeline_result.segments, pipeline_result.segment_count);
  out.output_text =
      format_segments(pipeline_result.segments, pipeline_result.segment_count);
  out.note =
      "profile=public_pipeline proof_status=maintained_loader_real_audio";
  out.note += " load_strategy=";
  out.note +=
      emel::tools::model_load_io_strategy_name(model.load.used_io_strategy);
  out.note += " pipeline_ns=" +
              std::to_string(static_cast<std::uint64_t>(pipeline_ns));
  out.note += " segment_checksum=" + std::to_string(out.output_checksum);
  return out;
}

} // namespace

void append_emel_sortformer_diarization_cases(std::vector<result> &results,
                                              const config &cfg) {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::pcm_fixture pcm{};
  fixture::expected_output_baseline baseline{};
  const double prepare_ns = measure_once_ns([&]() {
    require_model_fixture(*model);
    require_pcm_fixture(pcm);
    require_baseline(baseline);
    maybe_write_onnx_probe_inputs(*model, pcm);
  });
  fixture::run_result pipeline_result{};
  auto machine = std::make_unique<pipeline::sm>();

  const auto bench_cfg = pipeline_benchmark_config(cfg);
  auto measured = measure_case(fixture::k_case_name, bench_cfg, [&]() {
    g_stage_ok_sink = fixture::run_pipeline(*machine, model->contract, pcm.pcm,
                                            pcm.sample_rate, pipeline_result);
    g_checksum_sink = fixture::compute_checksum(pipeline_result.segments,
                                                pipeline_result.segment_count);
  });
  if (!g_stage_ok_sink ||
      pipeline_result.err != emel::error::cast(pipeline::error::none) ||
      pipeline_result.segment_count <= 0) {
    fail_sortformer_setup("steady_state_pipeline_run");
  }
  write_float_probe_file(std::getenv(k_probability_probe_env),
                         pipeline_result.probabilities);
  const std::uint64_t checksum = fixture::compute_checksum(
      pipeline_result.segments, pipeline_result.segment_count);
  auto record = with_sortformer_metadata(
      std::move(measured), "emel", k_emel_backend_id, "cpp", checksum,
      static_cast<std::uint64_t>(pipeline_result.segment_count),
      k_proof_status);
  record.note += " load_strategy=";
  record.note +=
      emel::tools::model_load_io_strategy_name(model->load.used_io_strategy);
  record.prepare_ns_per_op = prepare_ns;
  record.encode_ns_per_op = record.ns_per_op;
  record.publish_ns_per_op = 0.0;
  record.output_text =
      format_segments(pipeline_result.segments, pipeline_result.segment_count);
  results.push_back(std::move(record));
  results.push_back(
      make_profile_emel_result(*model, pcm, baseline, prepare_ns));
  const char *stage_profile = std::getenv(k_stage_profile_env);
  if (stage_profile != nullptr && stage_profile[0] != '\0') {
    results.push_back(make_stage_profile_result(*model, pcm, baseline));
  }
}

void append_reference_sortformer_diarization_cases(std::vector<result> &results,
                                                   const config &cfg) {
  fixture::expected_output_baseline baseline{};
  require_baseline(baseline);

  const auto bench_cfg = pipeline_benchmark_config(cfg);
  auto measured = measure_case(fixture::k_case_name, bench_cfg, [&]() {
    g_checksum_sink = baseline.output_checksum;
  });
  auto record = with_sortformer_metadata(
      std::move(measured), "reference", k_reference_backend_id, "recorded",
      baseline.output_checksum,
      static_cast<std::uint64_t>(baseline.segment_count),
      "recorded_maintained_baseline");
  record.prepare_ns_per_op = 0.0;
  record.encode_ns_per_op = 0.0;
  record.publish_ns_per_op = 0.0;
  results.push_back(std::move(record));
  results.push_back(make_profile_reference_result("reference", baseline));
}

} // namespace emel::bench
