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

#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"
#include "emel/diarization/sortformer/executor/sm.hpp"
#include "emel/diarization/sortformer/output/detail.hpp"
#include "diarization/sortformer_fixture.hpp"

namespace emel::bench {

namespace {

namespace fixture = emel::bench::diarization::sortformer_fixture;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace pipeline_detail = emel::diarization::sortformer::pipeline::detail;

constexpr const char * k_proof_status = "maintained_loader_real_audio";
constexpr const char * k_emel_backend_id = "emel.diarization.sortformer";
constexpr const char * k_reference_backend_id = "recorded.diarization.baseline";
constexpr const char * k_onnx_feature_input_env = "EMEL_DIARIZATION_ONNX_FEATURE_INPUT";
constexpr const char * k_onnx_encoder_probe_env = "EMEL_DIARIZATION_ONNX_ENCODER_PROBE_OUTPUT";
constexpr const char * k_hidden_probe_env = "EMEL_DIARIZATION_HIDDEN_PROBE_OUTPUT";
constexpr const char * k_probability_probe_env = "EMEL_DIARIZATION_PROBABILITY_PROBE_OUTPUT";
constexpr const char * k_stage_profile_env = "EMEL_DIARIZATION_STAGE_PROFILE";
volatile std::uint64_t g_checksum_sink = 0;
volatile bool g_stage_ok_sink = false;

[[noreturn]] void fail_sortformer_setup(const char * step) {
  std::fprintf(stderr,
               "error: sortformer diarization bench setup failed at %s "
               "(model=%s audio=%s baseline=%s)\n",
               step,
               fixture::k_model_rel_path.data(),
               fixture::k_audio_rel_path.data(),
               fixture::k_baseline_rel_path.data());
  std::exit(1);
}

void require_model_fixture(fixture::model_fixture & model) {
  if (!fixture::prepare(model)) {
    fail_sortformer_setup("prepare_model_fixture");
  }
}

void require_pcm_fixture(fixture::pcm_fixture & pcm) {
  if (!fixture::prepare(pcm)) {
    fail_sortformer_setup("prepare_pcm_fixture");
  }
}

void require_baseline(fixture::expected_output_baseline & baseline) {
  if (!fixture::prepare(baseline)) {
    fail_sortformer_setup("prepare_expected_output_baseline");
  }
}

void write_float_probe_file(const char * output_path, std::span<const float> values) {
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

void maybe_write_onnx_probe_inputs(const fixture::model_fixture & model,
                                   const fixture::pcm_fixture & pcm) {
  const char * feature_path = std::getenv(k_onnx_feature_input_env);
  const char * encoder_path = std::getenv(k_onnx_encoder_probe_env);
  if ((feature_path == nullptr || feature_path[0] == '\0') &&
      (encoder_path == nullptr || encoder_path[0] == '\0')) {
    return;
  }

  namespace encoder_detail = emel::diarization::sortformer::encoder::detail;
  namespace feature_detail = emel::diarization::sortformer::encoder::feature_extractor::detail;
  std::vector<float> features(static_cast<size_t>(feature_detail::k_required_feature_count));
  const auto feature_contract = feature_detail::make_contract(model.model);
  if (!feature_detail::contract_valid(feature_contract)) {
    fail_sortformer_setup("onnx_feature_contract");
  }
  feature_detail::compute(pcm.pcm, feature_contract, features);
  write_float_probe_file(feature_path, features);

  if (encoder_path == nullptr || encoder_path[0] == '\0') {
    return;
  }
  encoder_detail::contract encoder_contract = {};
  encoder_detail::pre_encoder_workspace workspace = {};
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));
  if (!encoder_detail::bind_contract(model.model, encoder_contract)) {
    fail_sortformer_setup("onnx_encoder_contract");
  }
  if (!encoder_detail::compute_encoder_frames_from_features(features,
                                                           encoder_contract,
                                                           workspace,
                                                           encoder_frames)) {
    fail_sortformer_setup("onnx_encoder_compute");
  }
  write_float_probe_file(encoder_path, encoder_frames);

  const char * hidden_path = std::getenv(k_hidden_probe_env);
  if (hidden_path == nullptr || hidden_path[0] == '\0') {
    return;
  }
  std::vector<float> hidden_frames(static_cast<size_t>(
      emel::diarization::sortformer::executor::detail::k_required_hidden_value_count));
  int32_t hidden_frame_count = 0;
  int32_t hidden_dim = 0;
  emel::error::type err =
      emel::diarization::sortformer::executor::detail::to_error(
          emel::diarization::sortformer::executor::error::none);
  emel::diarization::sortformer::executor::sm executor{};
  emel::diarization::sortformer::executor::event::execute request{
      model.contract,
      encoder_frames,
      hidden_frames,
      hidden_frame_count,
      hidden_dim,
  };
  request.error_out = &err;
  if (!executor.process_event(request) ||
      err != emel::diarization::sortformer::executor::detail::to_error(
                 emel::diarization::sortformer::executor::error::none) ||
      hidden_frame_count != emel::diarization::sortformer::executor::detail::k_frame_count ||
      hidden_dim != emel::diarization::sortformer::executor::detail::k_hidden_dim) {
    fail_sortformer_setup("onnx_hidden_compute");
  }
  write_float_probe_file(hidden_path, hidden_frames);
}

config pipeline_benchmark_config(const config & cfg) noexcept {
  config out = cfg;
  out.iterations = std::max<std::uint64_t>(1u, cfg.iterations);
  out.runs = std::max<std::size_t>(1u, cfg.runs);
  return out;
}

result with_sortformer_metadata(result out,
                                const char * lane,
                                const char * backend_id,
                                const char * backend_language,
                                const std::uint64_t checksum,
                                const std::uint64_t output_dim,
                                const char * proof_status) {
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
  out.note = std::string{fixture::k_profile_id} + " proof_status=" + proof_status;
  return out;
}

std::string format_segments(std::span<const fixture::output_detail::segment_record> segments,
                            const int32_t segment_count) {
  std::string out = {};
  for (int32_t i = 0; i < segment_count; ++i) {
    const auto & segment = segments[static_cast<size_t>(i)];
    out += "speaker=" + std::to_string(segment.speaker);
    out += " start_frame=" + std::to_string(segment.start_frame);
    out += " end_frame=" + std::to_string(segment.end_frame);
    out += '\n';
  }
  return out;
}

template <class fn_type>
double measure_once_ns(fn_type && fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

result make_profile_reference_result(const char * lane,
                                     const fixture::expected_output_baseline & baseline) {
  result out;
  out.name = fixture::k_profile_case_name;
  out.compare_group = fixture::k_profile_case_name;
  out.lane = lane;
  out.backend_id = std::string_view{lane} == "emel" ? k_emel_backend_id : k_reference_backend_id;
  out.backend_language = std::string_view{lane} == "emel" ? "cpp" : "recorded";
  out.comparison_mode = "measurement";
  out.model_id = fixture::k_model_id;
  out.fixture_id = fixture::k_fixture_id;
  out.workload_id = "diarization_sortformer_pipeline_profile_v1";
  out.comparable = false;
  out.iterations = 1;
  out.runs = 1;
  out.output_dim = static_cast<std::uint64_t>(baseline.segment_count);
  out.output_checksum = baseline.output_checksum;
  out.note =
      "profile=orchestrated_pipeline source=recorded_maintained_baseline "
      "proof_status=measurement_only";
  return out;
}

result make_profile_emel_result(const fixture::model_fixture & model,
                                const fixture::pcm_fixture & pcm,
                                const fixture::expected_output_baseline & baseline,
                                const double prepare_ns) {
  fixture::run_result pipeline_result{};
  pipeline::sm machine{};

  const double pipeline_ns = measure_once_ns([&]() {
    g_stage_ok_sink = fixture::run_pipeline(machine,
                                            model.contract,
                                            pcm.pcm,
                                            pcm.sample_rate,
                                            pipeline_result);
    g_checksum_sink =
        fixture::compute_checksum(pipeline_result.segments, pipeline_result.segment_count);
  });

  if (!g_stage_ok_sink ||
      pipeline_result.err != pipeline_detail::to_error(pipeline::error::none) ||
      pipeline_result.segment_count <= 0) {
    fail_sortformer_setup("profile_pipeline_run");
  }

  result out = make_profile_reference_result("emel", baseline);
  out.backend_id = k_emel_backend_id;
  out.backend_language = "cpp";
  out.comparison_mode = "measurement";
  out.ns_per_op = pipeline_ns;
  out.prepare_ns_per_op = prepare_ns;
  out.encode_ns_per_op = pipeline_ns;
  out.publish_ns_per_op = 0.0;
  out.output_dim = static_cast<std::uint64_t>(pipeline_result.segment_count);
  out.output_checksum = fixture::compute_checksum(pipeline_result.segments,
                                                  pipeline_result.segment_count);
  out.output_text = format_segments(pipeline_result.segments, pipeline_result.segment_count);
  out.note += " end_to_end_ns=" + std::to_string(static_cast<std::uint64_t>(out.ns_per_op));
  out.note += " prepare_ns=" + std::to_string(static_cast<std::uint64_t>(prepare_ns));
  out.note += " runtime_error=" + std::to_string(static_cast<std::uint32_t>(pipeline_result.err));
  return out;
}

result make_stage_profile_result(const fixture::model_fixture & model,
                                 const fixture::pcm_fixture & pcm,
                                 const fixture::expected_output_baseline & baseline) {
  namespace encoder_detail = emel::diarization::sortformer::encoder::detail;
  namespace feature_detail = emel::diarization::sortformer::encoder::feature_extractor::detail;
  namespace output_detail = emel::diarization::sortformer::output::detail;

  std::vector<float> features(static_cast<size_t>(feature_detail::k_required_feature_count));
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));
  std::vector<float> hidden_frames(static_cast<size_t>(
      emel::diarization::sortformer::executor::detail::k_required_hidden_value_count));
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count));
  std::array<output_detail::segment_record, pipeline_detail::k_max_segment_count> segments = {};

  const auto feature_contract = feature_detail::make_contract(model.model);
  encoder_detail::contract encoder_contract = {};
  emel::diarization::sortformer::modules::detail::contract modules_contract = {};
  encoder_detail::pre_encoder_workspace encoder_workspace = {};
  emel::diarization::sortformer::executor::sm executor = {};
  int32_t hidden_frame_count = 0;
  int32_t hidden_dim = 0;
  int32_t segment_count = 0;
  emel::error::type executor_err =
      emel::diarization::sortformer::executor::detail::to_error(
          emel::diarization::sortformer::executor::error::none);

  if (!feature_detail::contract_valid(feature_contract) ||
      !encoder_detail::bind_contract(model.model, encoder_contract) ||
      !emel::diarization::sortformer::modules::detail::bind_contract(model.model,
                                                                     modules_contract)) {
    fail_sortformer_setup("stage_profile_contract");
  }

  const double feature_ns = measure_once_ns([&]() {
    feature_detail::compute(pcm.pcm, feature_contract, features);
  });
  const double encoder_ns = measure_once_ns([&]() {
    g_stage_ok_sink = encoder_detail::compute_encoder_frames_from_features(features,
                                                                           encoder_contract,
                                                                           encoder_workspace,
                                                                           encoder_frames);
  });
  if (!g_stage_ok_sink) {
    fail_sortformer_setup("stage_profile_encoder");
  }

  const double executor_ns = measure_once_ns([&]() {
    emel::diarization::sortformer::executor::event::execute request{
        model.contract,
        encoder_frames,
        hidden_frames,
        hidden_frame_count,
        hidden_dim,
    };
    request.error_out = &executor_err;
    g_stage_ok_sink = executor.process_event(request);
  });
  if (!g_stage_ok_sink ||
      executor_err != emel::diarization::sortformer::executor::detail::to_error(
                          emel::diarization::sortformer::executor::error::none)) {
    fail_sortformer_setup("stage_profile_executor");
  }

  const double probabilities_ns = measure_once_ns([&]() {
    g_stage_ok_sink = output_detail::compute_speaker_probabilities(hidden_frames,
                                                                   modules_contract,
                                                                   probabilities);
  });
  if (!g_stage_ok_sink) {
    fail_sortformer_setup("stage_profile_probabilities");
  }

  const double decode_ns = measure_once_ns([&]() {
    g_stage_ok_sink = output_detail::decode_segments(probabilities,
                                                     output_detail::k_default_activity_threshold,
                                                     segments,
                                                     segment_count);
  });
  if (!g_stage_ok_sink || segment_count <= 0) {
    fail_sortformer_setup("stage_profile_decode");
  }

  result out = make_profile_reference_result("emel", baseline);
  out.name = "diarization/sortformer/stage_profile_ami_en2002b_mix_headset_137.00_152.04_16khz_mono";
  out.compare_group = out.name;
  out.backend_id = k_emel_backend_id;
  out.backend_language = "cpp";
  out.comparison_mode = "measurement";
  out.comparable = false;
  out.workload_id = "diarization_sortformer_stage_profile_v1";
  out.ns_per_op = feature_ns + encoder_ns + executor_ns + probabilities_ns + decode_ns;
  out.prepare_ns_per_op = feature_ns;
  out.encode_ns_per_op = encoder_ns + executor_ns;
  out.publish_ns_per_op = probabilities_ns + decode_ns;
  out.output_dim = static_cast<std::uint64_t>(segment_count);
  out.output_checksum = fixture::compute_checksum(segments, segment_count);
  out.output_text = format_segments(segments, segment_count);
  out.note = "profile=stage_breakdown proof_status=maintained_loader_real_audio";
  out.note += " feature_ns=" + std::to_string(static_cast<std::uint64_t>(feature_ns));
  out.note += " encoder_ns=" + std::to_string(static_cast<std::uint64_t>(encoder_ns));
  out.note += " executor_ns=" + std::to_string(static_cast<std::uint64_t>(executor_ns));
  out.note += " probabilities_ns=" +
      std::to_string(static_cast<std::uint64_t>(probabilities_ns));
  out.note += " decode_ns=" + std::to_string(static_cast<std::uint64_t>(decode_ns));
  out.note += " segment_checksum=" + std::to_string(out.output_checksum);
  return out;
}

}  // namespace

void append_emel_sortformer_diarization_cases(std::vector<result> & results,
                                              const config & cfg) {
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
  pipeline::sm machine{};

  const auto bench_cfg = pipeline_benchmark_config(cfg);
  auto measured = measure_case(fixture::k_case_name, bench_cfg, [&]() {
    g_stage_ok_sink = fixture::run_pipeline(machine,
                                            model->contract,
                                            pcm.pcm,
                                            pcm.sample_rate,
                                            pipeline_result);
    g_checksum_sink = fixture::compute_checksum(pipeline_result.segments, pipeline_result.segment_count);
  });
  if (!g_stage_ok_sink || pipeline_result.err != pipeline_detail::to_error(pipeline::error::none) ||
      pipeline_result.segment_count <= 0) {
    fail_sortformer_setup("steady_state_pipeline_run");
  }
  write_float_probe_file(std::getenv(k_probability_probe_env), pipeline_result.probabilities);
  const std::uint64_t checksum = fixture::compute_checksum(pipeline_result.segments,
                                                           pipeline_result.segment_count);
  auto record = with_sortformer_metadata(std::move(measured),
                                         "emel",
                                         k_emel_backend_id,
                                         "cpp",
                                         checksum,
                                         static_cast<std::uint64_t>(pipeline_result.segment_count),
                                         k_proof_status);
  record.prepare_ns_per_op = prepare_ns;
  record.encode_ns_per_op = record.ns_per_op;
  record.publish_ns_per_op = 0.0;
  record.output_text = format_segments(pipeline_result.segments, pipeline_result.segment_count);
  results.push_back(std::move(record));
  results.push_back(make_profile_emel_result(*model, pcm, baseline, prepare_ns));
  const char * stage_profile = std::getenv(k_stage_profile_env);
  if (stage_profile != nullptr && stage_profile[0] != '\0') {
    results.push_back(make_stage_profile_result(*model, pcm, baseline));
  }
}

void append_reference_sortformer_diarization_cases(std::vector<result> & results,
                                                   const config & cfg) {
  fixture::expected_output_baseline baseline{};
  require_baseline(baseline);

  const auto bench_cfg = pipeline_benchmark_config(cfg);
  auto measured = measure_case(fixture::k_case_name, bench_cfg, [&]() {
    g_checksum_sink = baseline.output_checksum;
  });
  auto record = with_sortformer_metadata(std::move(measured),
                                         "reference",
                                         k_reference_backend_id,
                                         "recorded",
                                         baseline.output_checksum,
                                         static_cast<std::uint64_t>(baseline.segment_count),
                                         "recorded_maintained_baseline");
  record.prepare_ns_per_op = 0.0;
  record.encode_ns_per_op = 0.0;
  record.publish_ns_per_op = 0.0;
  results.push_back(std::move(record));
  results.push_back(make_profile_reference_result("reference", baseline));
}

}  // namespace emel::bench
