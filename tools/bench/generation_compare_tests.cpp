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

std::filesystem::path generation_compare_script_path() {
#ifdef GENERATION_COMPARE_SCRIPT_PATH
  return GENERATION_COMPARE_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "generation_compare.py";
#endif
}

std::filesystem::path bench_generation_compare_wrapper_path() {
  return repo_root() / "scripts" / "bench_generation_compare.sh";
}

std::filesystem::path bench_generation_reference_wrapper_path() {
  return repo_root() / "scripts" / "bench_generation_reference_llama_cpp.sh";
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

void write_binary_file(const std::filesystem::path & path, const std::string & bytes) {
  std::ofstream output(path, std::ios::binary);
  REQUIRE(output.good());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(output.good());
}

void make_executable(const std::filesystem::path & path) {
  std::error_code ec = {};
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                 std::filesystem::perms::owner_write |
                                 std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::replace,
                               ec);
  REQUIRE(!ec);
}

std::string qwen_compare_record_json(const std::string & lane,
                                     const std::string & formatter_contract,
                                     const std::string & sampling_id,
                                     const int max_output_tokens,
                                     const std::string & output_path = "",
                                     const int output_tokens = 1,
                                     const int output_bytes = 5,
                                     const int output_checksum = 123) {
  const std::string backend_id =
    lane == "emel" ? "emel.generator" : "cpp.reference.llama_cpp";
  return "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
         "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
         "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"" +
         lane +
         "\","
         "\"backend_id\":\"" +
         backend_id +
         "\",\"backend_language\":\"cpp\","
         "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
         "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
         "qwen3_single_user_hello_max_tokens_1.json\","
         "\"comparison_mode\":\"parity\",\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\","
         "\"prompt_fixture_id\":\"single_user_hello_v1\","
         "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
         "\"prompt_id\":\"single_user:hello\","
         "\"formatter_mode\":\"chat_template_supported_qwen_v1\","
         "\"formatter_contract\":\"" +
         formatter_contract +
         "\",\"sampling_id\":\"" +
         sampling_id +
         "\","
         "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,"
         "\"max_output_tokens\":" +
         std::to_string(max_output_tokens) +
         ","
         "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
         "\"publish_ns_per_op\":0.1,\"output_tokens\":" +
         std::to_string(output_tokens) +
         ",\"output_bytes\":" +
         std::to_string(output_bytes) +
         ","
         "\"output_checksum\":" +
         std::to_string(output_checksum) +
         ",\"iterations\":1,\"runs\":1,\"output_path\":\"" +
         output_path +
         "\","
         "\"note\":\"\",\"error_kind\":\"\",\"error_message\":\"\"}\n";
}

}  // namespace

TEST_CASE("generation compare reports exact matches from canonical generation records") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "exact-match";
  const std::filesystem::path emel_output = tmp_dir / "emel.txt";
  const std::filesystem::path reference_output = tmp_dir / "reference.txt";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_output, "hello");
  write_text_file(reference_output, "hello");
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"emel\","
    "\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "qwen3_single_user_hello_max_tokens_1.json\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\","
    "\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_qwen_v1\","
    "\"formatter_contract\":\"chat_template_supported_qwen_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":5,"
    "\"output_checksum\":123,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_output.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    reference_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"reference\","
    "\"backend_id\":\"cpp.reference.llama_cpp\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "qwen3_single_user_hello_max_tokens_1.json\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\","
    "\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_qwen_v1\","
    "\"formatter_contract\":\"chat_template_supported_qwen_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":5,"
    "\"output_checksum\":123,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + reference_output.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"schema\": \"generation_compare_summary/v1\"") != std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"exact_match\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  CHECK(summary.find("\"exact_output_match\": true") != std::string::npos);
  CHECK(summary.find("\"exact_checksum_match\": true") != std::string::npos);
  CHECK(capture.stdout_text.find("status=exact_match reason=ok") != std::string::npos);
}

TEST_CASE("generation compare records reference backend build failures explicitly") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "build-failure";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path manifest_json = tmp_dir / "backend.json";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(
    emel_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"emel\","
    "\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "qwen3_single_user_hello_max_tokens_1.json\",\"comparison_mode\":\"parity\","
    "\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\",\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_qwen_v1\","
    "\"formatter_contract\":\"chat_template_supported_qwen_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":5,"
    "\"output_checksum\":123,\"iterations\":1,\"runs\":1,\"output_path\":\"\","
    "\"note\":\"\",\"error_kind\":\"\",\"error_message\":\"\"}\n");
  write_text_file(
    manifest_json,
    "{\n"
    "  \"id\": \"broken.reference\",\n"
    "  \"surface\": \"generation_compare/v1\",\n"
    "  \"language\": \"cpp\",\n"
    "  \"build_command\": [\"missing-build-tool\"],\n"
    "  \"run_command\": [\"missing-run-tool\"]\n"
    "}\n");

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --backend-manifest " + quote_arg_posix(manifest_json.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"error\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"reference_lane_error\"") != std::string::npos);
  const std::string reference_jsonl = read_file(output_dir / "raw" / "reference.jsonl");
  CHECK(reference_jsonl.find("\"record_type\": \"error\"") != std::string::npos);
  CHECK(reference_jsonl.find("\"error_kind\": \"missing_executable\"") != std::string::npos);
}

TEST_CASE("generation compare reports bounded drift without treating it as an operational failure") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "bounded-drift";
  const std::filesystem::path emel_output = tmp_dir / "emel.txt";
  const std::filesystem::path reference_output = tmp_dir / "reference.txt";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_output, "hello");
  write_text_file(reference_output, "hello!");
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"emel\","
    "\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "qwen3_single_user_hello_max_tokens_1.json\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\","
    "\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_qwen_v1\","
    "\"formatter_contract\":\"chat_template_supported_qwen_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":5,"
    "\"output_checksum\":123,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_output.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    reference_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/qwen3_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"qwen3/prompt_hello/max_tokens_1\",\"lane\":\"reference\","
    "\"backend_id\":\"cpp.reference.llama_cpp\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "qwen3_single_user_hello_max_tokens_1.json\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"qwen3\",\"fixture_id\":\"qwen3\","
    "\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_qwen_v1\","
    "\"formatter_contract\":\"chat_template_supported_qwen_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":true,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":6,"
    "\"output_checksum\":456,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + reference_output.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
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
  CHECK(summary.find("\"shared_prefix_bytes\": 5") != std::string::npos);
  CHECK(capture.stdout_text.find("status=bounded_drift reason=output_mismatch") !=
        std::string::npos);
}

TEST_CASE("generation compare reports empty comparable outputs as exact matches") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "empty-exact";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  qwen_compare_record_json("emel",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           "",
                                           0,
                                           0,
                                           0));
  write_text_file(reference_jsonl,
                  qwen_compare_record_json("reference",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           "",
                                           0,
                                           0,
                                           0));

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"exact_match\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"ok\"") != std::string::npos);
  CHECK(summary.find("\"exact_output_match\": true") != std::string::npos);
  CHECK(summary.find("\"exact_checksum_match\": false") != std::string::npos);
}

TEST_CASE("generation compare reports shared prefixes in UTF-8 bytes") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "utf8-prefix";
  const std::filesystem::path emel_output = tmp_dir / "emel.txt";
  const std::filesystem::path reference_output = tmp_dir / "reference.txt";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_output, "héllo");
  write_text_file(reference_output, "hé!");
  write_text_file(emel_jsonl,
                  qwen_compare_record_json("emel",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           emel_output.string(),
                                           1,
                                           6,
                                           123));
  write_text_file(reference_jsonl,
                  qwen_compare_record_json("reference",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           reference_output.string(),
                                           1,
                                           4,
                                           456));

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
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
  CHECK(summary.find("\"shared_prefix_bytes\": 3") != std::string::npos);
}

TEST_CASE("generation compare tolerates non-UTF8 output files") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "non-utf8";
  const std::filesystem::path emel_output = tmp_dir / "emel.bin";
  const std::filesystem::path reference_output = tmp_dir / "reference.bin";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::string invalid_output = "hello";
  invalid_output.push_back(static_cast<char>(0xFF));
  write_binary_file(emel_output, invalid_output);
  write_binary_file(reference_output, "hello!");
  write_text_file(emel_jsonl,
                  qwen_compare_record_json("emel",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           emel_output.string(),
                                           1,
                                           6,
                                           123));
  write_text_file(reference_jsonl,
                  qwen_compare_record_json("reference",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1,
                                           reference_output.string(),
                                           1,
                                           6,
                                           456));

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
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
  CHECK(summary.find("\"shared_prefix_bytes\": 5") != std::string::npos);
}

TEST_CASE("generation compare publishes single-lane workloads as non-comparable") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "non-comparable";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(
    emel_jsonl,
    "{\"schema\":\"generation_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"generation/preloaded_request/gemma4_prompt_hello_max_tokens_1\","
    "\"compare_group\":\"gemma4/prompt_hello/max_tokens_1\",\"lane\":\"emel\","
    "\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"workload_id\":\"gemma4_single_user_hello_max_tokens_1_v1\","
    "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
    "gemma4_single_user_hello_max_tokens_1.json\","
    "\"comparison_mode\":\"single_lane\",\"model_id\":\"gemma4\",\"fixture_id\":\"gemma4\","
    "\"prompt_fixture_id\":\"single_user_hello_v1\","
    "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\","
    "\"prompt_id\":\"single_user:hello\",\"formatter_mode\":\"chat_template_supported_v1\","
    "\"formatter_contract\":\"chat_template_supported_v1\",\"sampling_id\":\"argmax_v1\","
    "\"stop_id\":\"max_tokens_v1\",\"seed\":0,\"comparable\":false,\"max_output_tokens\":1,"
    "\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,"
    "\"publish_ns_per_op\":0.1,\"output_tokens\":1,\"output_bytes\":5,"
    "\"output_checksum\":123,\"iterations\":1,\"runs\":1,\"output_path\":\"\","
    "\"note\":\"reference_lane_unavailable_for_maintained_compare_surface\","
    "\"error_kind\":\"\",\"error_message\":\"\"}\n");
  write_text_file(reference_jsonl, "");

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"non_comparable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"single_lane_emel_workload\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  CHECK(capture.stdout_text.find("status=non_comparable reason=single_lane_emel_workload") !=
        std::string::npos);
}

TEST_CASE("generation compare rejects sampling contract mismatches as non-comparable") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" /
    "sampling-mismatch";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  qwen_compare_record_json("emel",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1));
  write_text_file(reference_jsonl,
                  qwen_compare_record_json("reference",
                                           "chat_template_supported_qwen_v1",
                                           "temperature_0_8_v1",
                                           1));

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"non_comparable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"sampling_id_mismatch\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
}

TEST_CASE("generation compare rejects formatter and token budget mismatches") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" /
    "formatter-budget-mismatch";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path reference_jsonl = tmp_dir / "reference.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  write_text_file(emel_jsonl,
                  qwen_compare_record_json("emel",
                                           "chat_template_supported_qwen_v1",
                                           "argmax_v1",
                                           1));
  write_text_file(reference_jsonl,
                  qwen_compare_record_json("reference",
                                           "chat_template_supported_qwen_v2",
                                           "argmax_v1",
                                           2));

  const std::string command =
    "python3 " + quote_arg_posix(generation_compare_script_path().string()) +
    " --emel-input " + quote_arg_posix(emel_jsonl.string()) +
    " --reference-input " + quote_arg_posix(reference_jsonl.string()) +
    " --output-dir " + quote_arg_posix(output_dir.string()) +
    " > " + quote_arg_posix(stdout_path.string()) +
    " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"non_comparable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"formatter_contract_mismatch\"") != std::string::npos);
  CHECK(summary.find("\"max_output_tokens\": 1") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
}

#if !defined(_WIN32)
TEST_CASE("generation compare wrapper honors --skip-emel-build without requiring cmake or ninja") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "skip-build";
  const std::filesystem::path fake_bin_dir = tmp_dir / "bin";
  const std::filesystem::path fake_python = fake_bin_dir / "python3";
  const std::filesystem::path fake_dirname = fake_bin_dir / "dirname";
  const std::filesystem::path invoked_path = tmp_dir / "python-invoked.txt";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path output_dir = tmp_dir / "out";
  std::filesystem::create_directories(fake_bin_dir);

  write_text_file(fake_python,
                  "#!/bin/sh\n"
                  "printf 'python-invoked mode=%s\\n' "
                  "\"$EMEL_GENERATION_REFERENCE_COMPILER_MODE\" > "
                  "\"$EMEL_TEST_INVOKED_PATH\"\n"
                  "exit 0\n");
  make_executable(fake_python);
  write_text_file(fake_dirname,
                  "#!/bin/sh\n"
                  "/usr/bin/dirname \"$@\"\n");
  make_executable(fake_dirname);

  std::string command;
  command = "PATH=" + quote_arg_posix(fake_bin_dir.string()) + " ";
  command += "EMEL_TEST_INVOKED_PATH=" + quote_arg_posix(invoked_path.string()) + " ";
  command += quote_arg_posix("/bin/bash") + " " +
    quote_arg_posix(bench_generation_compare_wrapper_path().string());
  command += " --reference-backend llama_cpp_generation --skip-emel-build --system --output-dir " +
    quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());

  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(read_file(invoked_path).find("python-invoked") != std::string::npos);
  CHECK(read_file(invoked_path).find("mode=system") != std::string::npos);
}

TEST_CASE("generation reference wrapper honors --run-only without requiring build tools") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "run-only";
  const std::filesystem::path fake_build_dir = tmp_dir / "build";
  const std::filesystem::path fake_runner = fake_build_dir / "bench_runner";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::filesystem::create_directories(fake_build_dir);

  write_text_file(fake_runner,
                  "#!/bin/sh\n"
                  "printf 'suite=%s mode=%s\\n' \"$EMEL_BENCH_SUITE\" \"$1\"\n");
  make_executable(fake_runner);

  std::string command;
  command = "EMEL_GENERATION_REFERENCE_BUILD_DIR=" + quote_arg_posix(fake_build_dir.string()) + " ";
  command += quote_arg_posix("/bin/bash") + " " +
    quote_arg_posix(bench_generation_reference_wrapper_path().string());
  command += " --run-only --system";
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());

  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("suite=generation mode=--mode=reference") != std::string::npos);
}

TEST_CASE("generation compare wrapper reproduces a maintained multi-engine workflow end to end") {
  const std::filesystem::path qwen_fixture =
    repo_root() / "tests" / "models" / "Qwen3-0.6B-Q8_0.gguf";
  const std::filesystem::path lfm2_fixture =
    repo_root() / "tests" / "models" / "LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
  if (!std::filesystem::exists(qwen_fixture) || !std::filesystem::exists(lfm2_fixture)) {
    return;
  }

  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "e2e";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::string command;
  command = "EMEL_BENCH_GENERATION_ITERS=1 ";
  command += "EMEL_BENCH_GENERATION_RUNS=1 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_RUNS=0 ";
  command += quote_arg_posix("/bin/bash") + " " +
    quote_arg_posix(bench_generation_compare_wrapper_path().string());
  command +=
    " --reference-backend llama_cpp_generation"
    " --workload-id lfm2_single_user_hello_max_tokens_1_v1"
    " --skip-emel-build --system --output-dir " +
    quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());

  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"schema\": \"generation_compare_summary/v1\"") != std::string::npos);
  CHECK(summary.find("\"id\": \"llama_cpp_generation\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  const bool saw_truthful_verdict =
    summary.find("\"comparison_status\": \"exact_match\"") != std::string::npos ||
    summary.find("\"comparison_status\": \"bounded_drift\"") != std::string::npos;
  CHECK(saw_truthful_verdict);
  const std::string emel_jsonl = read_file(output_dir / "raw" / "emel.jsonl");
  const std::string reference_jsonl = read_file(output_dir / "raw" / "reference.jsonl");
  CHECK(emel_jsonl.find("\"backend_id\":\"emel.generator\"") != std::string::npos);
  CHECK(reference_jsonl.find("\"backend_id\":\"cpp.reference.llama_cpp\"") != std::string::npos);
}

TEST_CASE("generation compare wrapper publishes maintained single-lane workflow end to end") {
  const std::filesystem::path lfm2_fixture =
    repo_root() / "tests" / "models" / "LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
  if (!std::filesystem::exists(lfm2_fixture)) {
    return;
  }

  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-generation-compare-tests" / "e2e-single-lane";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::string command;
  command = "EMEL_BENCH_GENERATION_ITERS=1 ";
  command += "EMEL_BENCH_GENERATION_RUNS=1 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_RUNS=0 ";
  command += quote_arg_posix("/bin/bash") + " " +
    quote_arg_posix(bench_generation_compare_wrapper_path().string());
  command +=
    " --reference-backend llama_cpp_generation"
    " --workload-id lfm2_single_user_hello_max_tokens_1_single_lane_v1"
    " --skip-emel-build --system --output-dir " +
    quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());

  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"non_comparable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"single_lane_emel_workload\"") != std::string::npos);
  CHECK(summary.find("\"failed\": false") != std::string::npos);
  const std::string emel_jsonl = read_file(output_dir / "raw" / "emel.jsonl");
  const std::string reference_jsonl = read_file(output_dir / "raw" / "reference.jsonl");
  CHECK(emel_jsonl.find("\"workload_id\":\"lfm2_single_user_hello_max_tokens_1_single_lane_v1\"") !=
        std::string::npos);
  CHECK(emel_jsonl.find("\"comparison_mode\":\"single_lane\"") != std::string::npos);
  CHECK(reference_jsonl.empty());
}
#endif
