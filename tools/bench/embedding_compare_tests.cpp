#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
  capture.exit_code = WEXITSTATUS(status);
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

TEST_CASE("python golden backend emits canonical compare records") {
  const std::filesystem::path tmp_dir =
    std::filesystem::temp_directory_path() / "emel-embedding-compare-tests" / "python-goldens";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path result_dir = tmp_dir / "vectors";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_EMBEDDING_BENCH_FORMAT=jsonl && ";
  command += "set EMEL_EMBEDDING_RESULT_DIR=" + quote_arg_windows(result_dir.string()) + " && ";
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
