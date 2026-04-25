#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::filesystem::path repo_root() {
#ifdef BENCH_REPO_ROOT
  return BENCH_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path diarization_compare_script_path() {
#ifdef DIARIZATION_COMPARE_SCRIPT_PATH
  return DIARIZATION_COMPARE_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "diarization_compare.py";
#endif
}

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string quote_arg_posix(const std::string & arg) {
  std::string out = "'";
  for (const char c : arg) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

struct process_capture {
  int exit_code = -1;
  std::string stdout_text = {};
  std::string stderr_text = {};
};

process_capture run_command_capture(const std::string & command,
                                    const std::filesystem::path & stdout_path,
                                    const std::filesystem::path & stderr_path) {
  process_capture capture = {};
  const int status = std::system(command.c_str());
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);
  std::error_code ec = {};
  std::filesystem::remove(stdout_path, ec);
  std::filesystem::remove(stderr_path, ec);
  if (status == -1) {
    return capture;
  }
#if defined(_WIN32)
  capture.exit_code = status;
#else
  if (WIFEXITED(status)) {
    capture.exit_code = WEXITSTATUS(status);
  } else {
    capture.exit_code = 1;
  }
#endif
  return capture;
}

void write_text_file(const std::filesystem::path & path, const std::string & text) {
  std::ofstream output(path);
  REQUIRE(output.good());
  output << text;
  REQUIRE(output.good());
}

std::string diarization_compare_record_json(const std::string & lane,
                                            const std::string & backend_id,
                                            const int output_dim,
                                            const int output_checksum,
                                            const std::string & output_path = "",
                                            const int output_bytes = 0) {
  return "{\"schema\":\"diarization_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
         "\"case_name\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
         "\"compare_group\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
         "\"lane\":\"" +
         lane +
         "\",\"backend_id\":\"" +
         backend_id +
         "\",\"backend_language\":\"cpp\","
         "\"comparison_mode\":\"parity\","
         "\"model_id\":\"diar_streaming_sortformer_4spk_v2_1_gguf\","
         "\"fixture_id\":\"ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
         "\"workload_id\":\"diarization_sortformer_pipeline_v1\","
         "\"comparable\":true,\"ns_per_op\":1.0,\"ns_min_per_op\":0.5,"
         "\"ns_mean_per_op\":1.25,\"ns_max_per_op\":2.0,\"prepare_ns_per_op\":0.1,"
         "\"encode_ns_per_op\":0.9,\"publish_ns_per_op\":0.0,"
         "\"output_bytes\":" +
         std::to_string(output_bytes) +
         ",\"output_dim\":" +
         std::to_string(output_dim) +
         ",\"output_checksum\":" +
         std::to_string(output_checksum) +
         ",\"iterations\":1,\"runs\":1,\"output_path\":\"" +
         output_path +
         "\",\"note\":\"proof_status=test\",\"error_kind\":\"\",\"error_message\":\"\"}\n";
}

}  // namespace

TEST_CASE("diarization compare reports exact matches from maintained checksum records") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" / "exact-match";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "recorded.diarization.baseline",
                                                  17,
                                                  1789550750));

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"schema\": \"diarization_compare_summary/v1\"") != std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"exact_match\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  CHECK(summary.find("\"exact_checksum_match\": true") != std::string::npos);
  CHECK(summary.find("\"exact_output_dim_match\": true") != std::string::npos);
  CHECK(summary.find("\"ns_min_per_op\": 0.5") != std::string::npos);
  CHECK(summary.find("\"ns_mean_per_op\": 1.25") != std::string::npos);
  CHECK(summary.find("\"ns_max_per_op\": 2.0") != std::string::npos);
  CHECK(capture.stdout_text.find("status=exact_match reason=ok") != std::string::npos);
}

TEST_CASE("diarization compare reports bounded drift when checksum or segment count changes") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" / "bounded-drift";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "recorded.diarization.baseline",
                                                  16,
                                                  1789550751));

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"bounded_drift\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  CHECK(capture.stdout_text.find("status=bounded_drift reason=output_mismatch") !=
        std::string::npos);
}

TEST_CASE("diarization compare reports each reference backend independently") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" /
      "multiple-reference-backends";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path onnx_reference_jsonl = tmp_dir / "onnx_reference.jsonl";
  const std::filesystem::path pytorch_reference_jsonl = tmp_dir / "pytorch_reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "recorded.diarization.baseline",
                                                  17,
                                                  1789550750));
  write_text_file(onnx_reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "onnx.sortformer.v2_1",
                                                  16,
                                                  1789550751));
  write_text_file(pytorch_reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "pytorch.nemo.sortformer.v2_1",
                                                  17,
                                                  1789550750));

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --onnx-reference-input " + quote_arg_posix(onnx_reference_jsonl.string()) +
      " --pytorch-reference-input " + quote_arg_posix(pytorch_reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"reference_backend_id\": \"recorded.diarization.baseline\"") !=
        std::string::npos);
  CHECK(summary.find("\"reference_backend_id\": \"onnx.sortformer.v2_1\"") !=
        std::string::npos);
  CHECK(summary.find("\"reference_backend_id\": \"pytorch.nemo.sortformer.v2_1\"") !=
        std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"exact_match\"") != std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"bounded_drift\"") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_backend=recorded.diarization.baseline") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("reference_backend=onnx.sortformer.v2_1") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("reference_backend=pytorch.nemo.sortformer.v2_1") !=
        std::string::npos);
}

TEST_CASE("diarization compare labels reference lane roles") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" / "lane-roles";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path pytorch_reference_jsonl = tmp_dir / "pytorch_reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl, "");
  write_text_file(pytorch_reference_jsonl,
                  "{\"schema\":\"diarization_compare/v1\",\"record_type\":\"result\","
                  "\"status\":\"ok\","
                  "\"case_name\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"compare_group\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"lane\":\"reference\",\"backend_id\":\"pytorch.nemo.sortformer.v2_1\","
                  "\"backend_language\":\"python/pytorch+nemo\",\"comparison_mode\":\"parity\","
                  "\"reference_role\":\"parity_reference\","
                  "\"model_id\":\"diar_streaming_sortformer_4spk_v2_1_gguf\","
                  "\"fixture_id\":\"ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"workload_id\":\"diarization_sortformer_pipeline_v1\","
                  "\"comparable\":true,\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.0,"
                  "\"encode_ns_per_op\":1.0,\"publish_ns_per_op\":0.0,"
                  "\"output_bytes\":0,\"output_dim\":17,\"output_checksum\":1789550750,"
                  "\"iterations\":1,\"runs\":1,\"output_path\":\"\","
                  "\"note\":\"proof_status=test\",\"error_kind\":\"\",\"error_message\":\"\"}\n");

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --pytorch-reference-input " + quote_arg_posix(pytorch_reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"reference_role\": \"parity_reference\"") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_role=parity_reference") != std::string::npos);
}

TEST_CASE("diarization compare fails parity reference drift") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" / "parity-drift";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path pytorch_reference_jsonl = tmp_dir / "pytorch_reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  71,
                                                  249612182));
  write_text_file(reference_jsonl, "");
  write_text_file(pytorch_reference_jsonl,
                  "{\"schema\":\"diarization_compare/v1\",\"record_type\":\"result\","
                  "\"status\":\"ok\","
                  "\"case_name\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"compare_group\":\"diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"lane\":\"reference\",\"backend_id\":\"pytorch.nemo.sortformer.v2_1\","
                  "\"backend_language\":\"python/pytorch+nemo\",\"comparison_mode\":\"parity\","
                  "\"reference_role\":\"parity_reference\","
                  "\"model_id\":\"diar_streaming_sortformer_4spk_v2_1_gguf\","
                  "\"fixture_id\":\"ami_en2002b_mix_headset_137.00_152.04_16khz_mono\","
                  "\"workload_id\":\"diarization_sortformer_pipeline_v1\","
                  "\"comparable\":true,\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.0,"
                  "\"encode_ns_per_op\":1.0,\"publish_ns_per_op\":0.0,"
                  "\"output_bytes\":0,\"output_dim\":17,\"output_checksum\":424967724,"
                  "\"iterations\":1,\"runs\":1,\"output_path\":\"\","
                  "\"note\":\"proof_status=test\",\"error_kind\":\"\",\"error_message\":\"\"}\n");

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --pytorch-reference-input " + quote_arg_posix(pytorch_reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"bounded_drift\"") != std::string::npos);
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_role=parity_reference") != std::string::npos);
}

TEST_CASE("diarization compare reports missing comparable records as failures") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" / "missing";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl, "");

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"missing\"") != std::string::npos);
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(capture.stdout_text.find("status=missing reason=missing_reference_record") !=
        std::string::npos);
}

TEST_CASE("diarization compare reports requested ONNX model errors as failures") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-diarization-compare-tests" /
      "missing-onnx-model";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path missing_model = tmp_dir / "missing.onnx";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  diarization_compare_record_json("emel",
                                                  "emel.diarization.sortformer",
                                                  17,
                                                  1789550750));
  write_text_file(reference_jsonl,
                  diarization_compare_record_json("reference",
                                                  "recorded.diarization.baseline",
                                                  17,
                                                  1789550750));

  const std::string command =
      "python3 " + quote_arg_posix(diarization_compare_script_path().string()) +
      " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
      " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
      " --onnx-reference-model " + quote_arg_posix(missing_model.string()) +
      " --output-dir " + quote_arg_posix(output_dir.string()) +
      " > " + quote_arg_posix(stdout_path.string()) +
      " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"reference_backend_id\": \"onnx.sortformer.v2_1\"") !=
        std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"error\"") != std::string::npos);
  CHECK(summary.find("\"error_kind\": \"missing_model\"") != std::string::npos);
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(capture.stdout_text.find("status=error reason=reference_lane_error") !=
        std::string::npos);
}
