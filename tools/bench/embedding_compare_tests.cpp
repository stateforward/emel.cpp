#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <csignal>
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

std::filesystem::path embedding_compare_script_path() {
#ifdef EMBEDDING_COMPARE_SCRIPT_PATH
  return EMBEDDING_COMPARE_SCRIPT_PATH;
#else
  return repo_root() / "tools" / "bench" / "embedding_compare.py";
#endif
}

std::filesystem::path embedding_reference_python_path() {
#ifdef EMBEDDING_REFERENCE_PYTHON_PATH
  return EMBEDDING_REFERENCE_PYTHON_PATH;
#else
  return repo_root() / "tools" / "bench" / "embedding_reference_python.py";
#endif
}

std::filesystem::path embedding_generator_bench_runner_path() {
#ifdef EMBEDDING_GENERATOR_BENCH_RUNNER_PATH
  return EMBEDDING_GENERATOR_BENCH_RUNNER_PATH;
#else
  return repo_root() / "build" / "bench_tools_ninja" / "embedding_generator_bench_runner";
#endif
}

std::filesystem::path bench_embedding_compare_wrapper_path() {
  return repo_root() / "scripts" / "bench_embedding_compare.sh";
}

std::filesystem::path bench_embedding_reference_liquid_wrapper_path() {
  return repo_root() / "scripts" / "bench_embedding_reference_liquid.sh";
}

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

size_t count_substring(std::string_view text, std::string_view needle) {
  if (needle.empty()) {
    return 0;
  }
  size_t count = 0;
  size_t position = 0;
  while ((position = text.find(needle, position)) != std::string_view::npos) {
    ++count;
    position += needle.size();
  }
  return count;
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

std::string quote_arg_windows(const std::string & arg) {
  std::string out = "\"";
  for (const char c : arg) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

std::string set_env_windows(const std::string & name, const std::string & value) {
  std::string out = "set \"";
  out += name;
  out += "=";
  for (const char c : value) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
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
  } else if (WIFSIGNALED(status)) {
    capture.exit_code = 128 + WTERMSIG(status);
    if (!capture.stderr_text.empty() && capture.stderr_text.back() != '\n') {
      capture.stderr_text.push_back('\n');
    }
    capture.stderr_text +=
      "process terminated by signal " + std::to_string(WTERMSIG(status)) + "\n";
  } else {
    capture.exit_code = 1;
    if (!capture.stderr_text.empty() && capture.stderr_text.back() != '\n') {
      capture.stderr_text.push_back('\n');
    }
    capture.stderr_text += "process exited with unknown wait status\n";
  }
#endif
  return capture;
}

void write_binary_floats(const std::filesystem::path & path, const std::initializer_list<float> values) {
  std::ofstream output(path, std::ios::binary);
  REQUIRE(output.good());
  for (const float value : values) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  REQUIRE(output.good());
}

void write_text_file(const std::filesystem::path & path, const std::string & text) {
  std::ofstream output(path);
  REQUIRE(output.good());
  output << text;
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

}  // namespace

TEST_CASE("embedding compare computes parity metrics from canonical records") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "parity";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_vector = tmp_dir / "emel.f32";
  const std::filesystem::path ref_vector = tmp_dir / "ref.f32";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  write_binary_floats(emel_vector, {1.0f, 0.0f, 0.0f});
  write_binary_floats(ref_vector, {0.5f, 0.5f, 0.0f});
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"emel/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"emel\",\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":3,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    ref_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"reference/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"reference\",\"backend_id\":\"python.reference.test\",\"backend_language\":\"python\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":3,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + ref_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"computed\"") != std::string::npos);
  CHECK(summary.find("\"cosine\"") != std::string::npos);
  CHECK(capture.stdout_text.find("text/red_square/full_dim status=computed") != std::string::npos);
}

TEST_CASE("embedding compare marks baseline backends as similarity unavailable") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "baseline";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_vector = tmp_dir / "emel.f32";
  const std::filesystem::path ref_vector = tmp_dir / "ref.f32";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  write_binary_floats(emel_vector, {1.0f, 2.0f});
  write_binary_floats(ref_vector, {3.0f, 4.0f});
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"emel/image\",\"compare_group\":\"image/red_square/full_dim\","
    "\"lane\":\"emel\",\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"image\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":2,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    ref_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"reference/image\",\"compare_group\":\"image/red_square/full_dim\","
    "\"lane\":\"reference\",\"backend_id\":\"cpp.reference.lfm2\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"baseline\",\"model_id\":\"lfm2\",\"fixture_id\":\"fixture\","
    "\"modality\":\"image\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":2,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + ref_vector.string() + "\",\"note\":\"baseline_compare_only\","
    "\"error_kind\":\"\",\"error_message\":\"\"}\n");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"unavailable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"non_parity_backend\"") != std::string::npos);
}

TEST_CASE("embedding compare preserves every reference record that shares a compare group") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "duplicate-groups";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_vector = tmp_dir / "emel.f32";
  const std::filesystem::path ref_vector_a = tmp_dir / "ref-a.f32";
  const std::filesystem::path ref_vector_b = tmp_dir / "ref-b.f32";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  write_binary_floats(emel_vector, {1.0f, 2.0f});
  write_binary_floats(ref_vector_a, {3.0f, 4.0f});
  write_binary_floats(ref_vector_b, {5.0f, 6.0f});
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"emel/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"emel\",\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":2,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    ref_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"reference/text/arctic_s\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"reference\",\"backend_id\":\"cpp.reference.arctic_s\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"baseline\",\"model_id\":\"arctic_s\",\"fixture_id\":\"fixture-a\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":2,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + ref_vector_a.string() + "\",\"note\":\"baseline_compare_only\","
    "\"error_kind\":\"\",\"error_message\":\"\"}\n"
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"reference/text/embeddinggemma_300m\","
    "\"compare_group\":\"text/red_square/full_dim\",\"lane\":\"reference\","
    "\"backend_id\":\"cpp.reference.embeddinggemma_300m\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"baseline\",\"model_id\":\"embeddinggemma_300m\","
    "\"fixture_id\":\"fixture-b\",\"modality\":\"text\",\"ns_per_op\":1.0,"
    "\"prepare_ns_per_op\":0.1,\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,"
    "\"output_tokens\":1,\"output_dim\":2,\"output_checksum\":2,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + ref_vector_b.string() + "\",\"note\":\"baseline_compare_only\","
    "\"error_kind\":\"\",\"error_message\":\"\"}\n");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"backend_id\": \"cpp.reference.arctic_s\"") != std::string::npos);
  CHECK(summary.find("\"backend_id\": \"cpp.reference.embeddinggemma_300m\"") != std::string::npos);
  CHECK(count_substring(summary, "\"reason\": \"non_parity_backend\"") == 2);
  CHECK(count_substring(capture.stdout_text,
                        "text/red_square/full_dim status=unavailable reason=non_parity_backend") ==
        2);
}

TEST_CASE("embedding compare fails when a compare group is missing a reference lane record") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "missing-reference";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_vector = tmp_dir / "emel.f32";
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  write_binary_floats(emel_vector, {1.0f, 0.0f, 0.0f});
  write_text_file(
    emel_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"emel/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"emel\",\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":3,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + emel_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(ref_jsonl, "");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(summary.find("\"comparison_status\": \"missing\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"missing_reference_record\"") != std::string::npos);
}

TEST_CASE("embedding compare fails when both lanes produce no compare groups") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "no-groups";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  write_text_file(emel_jsonl, "");
  write_text_file(ref_jsonl, "");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 1);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"failed\": true") != std::string::npos);
  CHECK(summary.find("\"groups\": []") != std::string::npos);
}

TEST_CASE("embedding compare reports missing output vector files without aborting") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "missing-vectors";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path emel_jsonl = tmp_dir / "emel.jsonl";
  const std::filesystem::path ref_jsonl = tmp_dir / "ref.jsonl";
  const std::filesystem::path output_dir = tmp_dir / "out";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path missing_emel_vector = tmp_dir / "missing-emel.f32";
  const std::filesystem::path missing_ref_vector = tmp_dir / "missing-ref.f32";

  write_text_file(
    emel_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"emel/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"emel\",\"backend_id\":\"emel.generator\",\"backend_language\":\"cpp\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":3,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + missing_emel_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");
  write_text_file(
    ref_jsonl,
    "{\"schema\":\"embedding_compare/v1\",\"record_type\":\"result\",\"status\":\"ok\","
    "\"case_name\":\"reference/text\",\"compare_group\":\"text/red_square/full_dim\","
    "\"lane\":\"reference\",\"backend_id\":\"python.reference.test\",\"backend_language\":\"python\","
    "\"comparison_mode\":\"parity\",\"model_id\":\"te\",\"fixture_id\":\"fixture\","
    "\"modality\":\"text\",\"ns_per_op\":1.0,\"prepare_ns_per_op\":0.1,"
    "\"encode_ns_per_op\":0.8,\"publish_ns_per_op\":0.1,\"output_tokens\":1,"
    "\"output_dim\":3,\"output_checksum\":1,\"iterations\":1,\"runs\":1,"
    "\"output_path\":\"" + missing_ref_vector.string() + "\",\"note\":\"\",\"error_kind\":\"\","
    "\"error_message\":\"\"}\n");

  std::string command;
#if defined(_WIN32)
  command = "python3 " + quote_arg_windows(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_windows(emel_jsonl.string());
  command += " --reference-input " + quote_arg_windows(ref_jsonl.string());
  command += " --output-dir " + quote_arg_windows(output_dir.string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "python3 " + quote_arg_posix(embedding_compare_script_path().string());
  command += " --emel-input " + quote_arg_posix(emel_jsonl.string());
  command += " --reference-input " + quote_arg_posix(ref_jsonl.string());
  command += " --output-dir " + quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  const std::string summary = read_file(output_dir / "compare_summary.json");
  CHECK(summary.find("\"comparison_status\": \"unavailable\"") != std::string::npos);
  CHECK(summary.find("\"reason\": \"missing_output_vectors\"") != std::string::npos);
}

TEST_CASE("embedding generator compare output survives warmup iterations") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" /
    "generator-warmup-output";
  const std::filesystem::path result_dir = tmp_dir / "vectors";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::string command;
#if defined(_WIN32)
  command = set_env_windows("EMEL_BENCH_ITERS", "1");
  command += " && " + set_env_windows("EMEL_BENCH_RUNS", "1");
  command += " && " + set_env_windows("EMEL_BENCH_WARMUP_ITERS", "1");
  command += " && " + set_env_windows("EMEL_BENCH_WARMUP_RUNS", "1");
  command += " && " + set_env_windows("EMEL_BENCH_CASE_FILTER", "text_red_square_full_dim");
  command += " && " + set_env_windows("EMEL_EMBEDDING_BENCH_FORMAT", "jsonl");
  command += " && " + set_env_windows("EMEL_EMBEDDING_RESULT_DIR", result_dir.string());
  command += " && " + quote_arg_windows(embedding_generator_bench_runner_path().string());
  command += " > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=1 ";
  command += "EMEL_BENCH_WARMUP_RUNS=1 ";
  command += "EMEL_BENCH_CASE_FILTER=text_red_square_full_dim ";
  command += "EMEL_EMBEDDING_BENCH_FORMAT=jsonl ";
  command += "EMEL_EMBEDDING_RESULT_DIR=" + quote_arg_posix(result_dir.string()) + " ";
  command += quote_arg_posix(embedding_generator_bench_runner_path().string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("warning: skipping embedding generator benchmark") ==
        std::string::npos);
  CHECK(capture.stdout_text.find("\"lane\":\"emel\"") != std::string::npos);
  CHECK(capture.stdout_text.find("\"output_path\":\"\"") == std::string::npos);

  std::size_t dumped_outputs = 0u;
  for (const auto & entry : std::filesystem::directory_iterator(result_dir)) {
    if (entry.path().extension() == ".f32") {
      ++dumped_outputs;
    }
  }
  CHECK(dumped_outputs == 1u);
}

#if !defined(_WIN32)
TEST_CASE("run_command_capture reports signal termination deterministically") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "signal-status";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
    "kill -TERM $$ > " + quote_arg_posix(stdout_path.string()) + " 2> " +
    quote_arg_posix(stderr_path.string());

  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 128 + SIGTERM);
  CHECK(capture.stderr_text.find("process terminated by signal") != std::string::npos);
}
#endif

#if !defined(_WIN32)
TEST_CASE("compare wrapper honors --skip-emel-build without requiring cmake or ninja") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "skip-build";
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
                  "printf 'python-invoked\\n' > \"$EMEL_TEST_INVOKED_PATH\"\n"
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
    quote_arg_posix(bench_embedding_compare_wrapper_path().string());
  command += " --reference-backend te_python_goldens --skip-emel-build --system --output-dir " +
    quote_arg_posix(output_dir.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(read_file(invoked_path).find("python-invoked") != std::string::npos);
}

TEST_CASE("liquid wrapper honors --run-only without requiring build tools") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "liquid-run-only";
  const std::filesystem::path fake_bin_dir = tmp_dir / "bin";
  const std::filesystem::path fake_dirname = fake_bin_dir / "dirname";
  const std::filesystem::path fake_mkdir = fake_bin_dir / "mkdir";
  const std::filesystem::path build_dir = tmp_dir / "build";
  const std::filesystem::path asset_dir = tmp_dir / "assets";
  const std::filesystem::path fake_runner = build_dir / "embedding_reference_bench_runner";
  const std::filesystem::path invoked_path = tmp_dir / "runner-invoked.txt";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::filesystem::create_directories(fake_bin_dir);
  std::filesystem::create_directories(build_dir);

  write_text_file(fake_dirname,
                  "#!/bin/sh\n"
                  "/usr/bin/dirname \"$@\"\n");
  make_executable(fake_dirname);
  write_text_file(fake_mkdir,
                  "#!/bin/sh\n"
                  "/bin/mkdir \"$@\"\n");
  make_executable(fake_mkdir);
  write_text_file(fake_runner,
                  "#!/bin/sh\n"
                  "printf 'runner-invoked\\n' > \"$EMEL_TEST_INVOKED_PATH\"\n"
                  "exit 0\n");
  make_executable(fake_runner);

  std::string command;
  command = "PATH=" + quote_arg_posix(fake_bin_dir.string()) + " ";
  command += "EMEL_TEST_INVOKED_PATH=" + quote_arg_posix(invoked_path.string()) + " ";
  command += "EMEL_REFERENCE_BUILD_DIR=" + quote_arg_posix(build_dir.string()) + " ";
  command += "EMEL_REFERENCE_ASSET_DIR=" + quote_arg_posix(asset_dir.string()) + " ";
  command += quote_arg_posix("/bin/bash") + " " +
    quote_arg_posix(bench_embedding_reference_liquid_wrapper_path().string());
  command += " --run-only --system";
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(read_file(invoked_path).find("runner-invoked") != std::string::npos);
}
#endif

TEST_CASE("python golden backend emits canonical compare records") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "python-goldens";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path result_dir = tmp_dir / "vectors";

  std::string command;
#if defined(_WIN32)
  command = set_env_windows("EMEL_EMBEDDING_BENCH_FORMAT", "jsonl") + " && ";
  command += set_env_windows("EMEL_EMBEDDING_RESULT_DIR", result_dir.string()) + " && ";
  command += "python3 " + quote_arg_windows(embedding_reference_python_path().string());
  command += " --backend te75m_goldens > " + quote_arg_windows(stdout_path.string());
  command += " 2> " + quote_arg_windows(stderr_path.string());
#else
  command = "EMEL_EMBEDDING_BENCH_FORMAT=jsonl ";
  command += "EMEL_EMBEDDING_RESULT_DIR=" + quote_arg_posix(result_dir.string()) + " ";
  command += "python3 " + quote_arg_posix(embedding_reference_python_path().string());
  command += " --backend te75m_goldens > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());
#endif
  const process_capture capture = run_command_capture(command, stdout_path, stderr_path);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
  CHECK(capture.stdout_text.find("\"backend_id\": \"python.reference.te75m_goldens\"") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("\"compare_group\": \"text/red_square/full_dim\"") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("\"compare_group\": \"image/red_square/full_dim\"") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("\"compare_group\": \"audio/pure_tone_440hz/full_dim\"") !=
        std::string::npos);
  CHECK(std::filesystem::exists(result_dir));
}
