#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#include "../generation_fixture_registry.hpp"
#include "generation_workload_manifest.hpp"

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

std::filesystem::path bench_runner_binary_path() {
#ifdef BENCH_RUNNER_BINARY_PATH
  return BENCH_RUNNER_BINARY_PATH;
#else
  return std::filesystem::path("bench_runner");
#endif
}

std::filesystem::path maintained_generation_fixture_path(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
  return repo_root() / fixture.fixture_rel;
}

bool maintained_generation_fixture_exists(
    const emel::tools::generation_fixture_registry::maintained_fixture & fixture) {
  return std::filesystem::exists(maintained_generation_fixture_path(fixture));
}

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
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

process_capture run_generation_bench_capture(const std::string & mode,
                                             const bool emit_jsonl = false) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      (emit_jsonl ? ("jsonl-" + mode) : ("text-" + mode));
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path output_dir = tmp_dir / "outputs";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_BENCH_SUITE=generation && ";
  command += "set EMEL_BENCH_ITERS=1 && ";
  command += "set EMEL_BENCH_RUNS=1 && ";
  command += "set EMEL_BENCH_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_WARMUP_RUNS=0 && ";
  command += "set EMEL_BENCH_GENERATION_ITERS=1 && ";
  command += "set EMEL_BENCH_GENERATION_RUNS=1 && ";
  command += "set EMEL_BENCH_GENERATION_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_GENERATION_WARMUP_RUNS=0 && ";
  if (emit_jsonl) {
    command += "set EMEL_GENERATION_BENCH_FORMAT=jsonl && ";
    command += "set \"EMEL_GENERATION_RESULT_DIR=";
    command += output_dir.string();
    command += "\" && ";
  }
  command += quote_arg_windows(bench_runner_binary_path().string());
  command += " --mode=" + mode + " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += "EMEL_BENCH_SUITE=generation ";
  command += "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_WARMUP_RUNS=0 ";
  command += "EMEL_BENCH_GENERATION_ITERS=1 ";
  command += "EMEL_BENCH_GENERATION_RUNS=1 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_GENERATION_WARMUP_RUNS=0 ";
  if (emit_jsonl) {
    command += "EMEL_GENERATION_BENCH_FORMAT=jsonl ";
    command += "EMEL_GENERATION_RESULT_DIR=" + quote_arg_posix(output_dir.string()) + " ";
  }
  command += quote_arg_posix(bench_runner_binary_path().string());
  command += " --mode=" + mode + " > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());
#endif

  const int status = std::system(command.c_str());
  process_capture capture{};
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);

  std::error_code ec;
  std::filesystem::remove(stdout_path, ec);
  std::filesystem::remove(stderr_path, ec);

  if (status == -1) {
    return capture;
  }
#if defined(_WIN32)
  capture.exit_code = status;
#else
  if (!WIFEXITED(status)) {
    return capture;
  }
  capture.exit_code = WEXITSTATUS(status);
#endif
  return capture;
}

process_capture run_generation_bench_compare_capture() {
  return run_generation_bench_capture("compare", false);
}

std::uint64_t parse_named_metric(const std::string & haystack, const std::string & name) {
  const std::string needle = name + "=";
  const size_t pos = haystack.find(needle);
  if (pos == std::string::npos) {
    return 0u;
  }

  size_t cursor = pos + needle.size();
  std::uint64_t value = 0u;
  while (cursor < haystack.size() && haystack[cursor] >= '0' && haystack[cursor] <= '9') {
    value = value * 10u + static_cast<std::uint64_t>(haystack[cursor] - '0');
    ++cursor;
  }
  return value;
}

std::string find_line_with_prefix(const std::string & haystack, const std::string & prefix) {
  const size_t pos = haystack.find(prefix);
  if (pos == std::string::npos) {
    return {};
  }

  const size_t line_end = haystack.find('\n', pos);
  if (line_end == std::string::npos) {
    return haystack.substr(pos);
  }
  return haystack.substr(pos, line_end - pos);
}
}  // namespace

TEST_CASE("bench_runner generation compare keeps maintained Qwen and Liquid fixtures") {
  const process_capture capture = run_generation_bench_compare_capture();
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("# generation_architecture: lfm2") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_formatter_contract:") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_stage_probe: case="
                                 "generation/preloaded_request/"
                                 "lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("emel_prefill_linear_probe_ns=") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_prefill_attention_probe_ns=") !=
        std::string::npos);

  bool saw_fixture = false;
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_maintained_generation_fixtures) {
    if (!maintained_generation_fixture_exists(fixture)) {
      continue;
    }
    saw_fixture = true;
    const std::array<int, 4> max_tokens = {1, 10, 100, 1000};
    for (const int tokens : max_tokens) {
      const std::string case_name = "generation/preloaded_request/" +
                                    std::string{fixture.slug} +
                                    "_prompt_hello_max_tokens_" +
                                    std::to_string(tokens);
      CHECK(capture.stdout_text.find(case_name) != std::string::npos);
    }
  }
  CHECK(saw_fixture);
  const std::string binary_size_line =
      find_line_with_prefix(capture.stdout_text, "# binary_size_compare:");
  CHECK_FALSE(binary_size_line.empty());
  CHECK(binary_size_line.find("status=ok") != std::string::npos);
  CHECK(parse_named_metric(binary_size_line, "emel_bytes") > 0u);
  CHECK(parse_named_metric(binary_size_line, "llama_bytes") > 0u);
}

TEST_CASE("bench_runner generation jsonl emits manifest-driven workload metadata and explicit comparability") {
  const process_capture emel_capture = run_generation_bench_capture("emel", true);
  CHECK(emel_capture.exit_code == 0);
  CHECK(emel_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"schema\":\"generation_compare/v1\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"emel\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"reference\"") == std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"backend_id\":\"emel.generator\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"backend_id\":\"cpp.reference.llama_cpp\"") ==
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_manifest_path\":\"tools/bench/generation_workloads/"
            "qwen3_single_user_hello_max_tokens_1.json\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"prompt_fixture_id\":\"single_user_hello_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"prompt_id\":\"single_user:hello\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"formatter_mode\":\"chat_template_supported_qwen_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"sampling_id\":\"argmax_v1\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparable\":true") != std::string::npos);
  const std::filesystem::path gemma4_fixture_path =
      repo_root() / "tests" / "models" / "gemma-4-e2b-it-Q8_0.gguf";
  if (std::filesystem::exists(gemma4_fixture_path)) {
    CHECK(emel_capture.stdout_text.find("\"comparison_mode\":\"single_lane\"") !=
          std::string::npos);
    CHECK(emel_capture.stdout_text.find(
              "\"note\":\"reference_lane_unavailable_for_maintained_compare_surface\"") !=
          std::string::npos);
  }
  CHECK(emel_capture.stdout_text.find("\"output_path\":\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("ns/op,") == std::string::npos);

  const process_capture reference_capture = run_generation_bench_capture("reference", true);
  CHECK(reference_capture.exit_code == 0);
  CHECK(reference_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"schema\":\"generation_compare/v1\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"reference\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"emel\"") == std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"backend_id\":\"cpp.reference.llama_cpp\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"backend_id\":\"emel.generator\"") ==
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"workload_manifest_path\":\"tools/bench/generation_workloads/") !=
        std::string::npos);
  const bool saw_supported_reference_formatter =
      reference_capture.stdout_text.find("\"formatter_mode\":\"chat_template_supported_qwen_v1\"") !=
          std::string::npos ||
      reference_capture.stdout_text.find("\"formatter_mode\":\"chat_template_supported_v1\"") !=
          std::string::npos;
  CHECK(saw_supported_reference_formatter);
  CHECK(reference_capture.stdout_text.find("\"comparable\":true") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"comparison_mode\":\"single_lane\"") ==
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"formatter_contract\":\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"output_path\":\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("ns/op,") == std::string::npos);
}

TEST_CASE("generation prompt fixture parser ignores quoted key names inside text values") {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" / "prompt-key-text";
  const std::filesystem::path prompt_path = tmp_dir / "prompt.json";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::ofstream output(prompt_path);
  REQUIRE(output.good());
  output << "{\n"
            "  \"schema\": \"generation_prompt_fixture/v1\",\n"
            "  \"id\": \"quoted_key_prompt_v1\",\n"
            "  \"shape\": \"single_user_text_v1\",\n"
            "  \"text\": \"literal marker \\\"prompt_id\\\" before metadata\",\n"
            "  \"prompt_id\": \"single_user:quoted_key\"\n"
            "}\n";
  REQUIRE(output.good());
  output.close();

  emel::bench::generation_prompt_fixture fixture = {};
  std::string error = {};
  CHECK(emel::bench::load_generation_prompt_fixture(prompt_path, fixture, &error));
  CHECK(error.empty());
  CHECK(fixture.text == "literal marker \"prompt_id\" before metadata");
  CHECK(fixture.prompt_id == "single_user:quoted_key");
}
