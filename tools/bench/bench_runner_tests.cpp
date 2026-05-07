#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "../generation_fixture_registry.hpp"
#include "bench_common.hpp"
#include "bench_dependency_manifest.hpp"
#include "bench_runner_contract.hpp"
#include "bench_runner_registry.hpp"
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

constexpr const char *k_bounded_generation_workload_id =
    "lfm2_single_user_hello_max_tokens_1_v1";
constexpr const char *k_bounded_generation_case_name =
    "generation/preloaded_request/"
    "lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1";

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

void write_file(const std::filesystem::path & path, const std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  REQUIRE(output);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  REQUIRE(output);
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

process_capture run_bench_runner_capture(const std::vector<std::string> & args,
                                         const std::string & tag) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" / tag;
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(bench_runner_binary_path().string());
  for (const std::string & arg : args) {
    command += " " + quote_arg_windows(arg);
  }
  command += " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(bench_runner_binary_path().string());
  for (const std::string & arg : args) {
    command += " " + quote_arg_posix(arg);
  }
  command += " > ";
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

process_capture run_serialized_request_capture(const std::string_view request_text,
                                               const std::string & tag,
                                               std::string & result_text,
                                               const bool enable_internal = false) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" / tag;
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path request_path = tmp_dir / "request.txt";
  const std::filesystem::path result_path = tmp_dir / "result.txt";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  write_file(request_path, request_text);

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_GENERATION_WORKLOAD_ID=";
  command += k_bounded_generation_workload_id;
  command += " && ";
  if (enable_internal) {
    command += "set EMEL_BENCH_INTERNAL=1 && ";
  }
  command += quote_arg_windows(bench_runner_binary_path().string());
  command += " --run-serialized-request ";
  command += quote_arg_windows(request_path.string());
  command += " --write-serialized-result ";
  command += quote_arg_windows(result_path.string());
  command += " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += "EMEL_GENERATION_WORKLOAD_ID=";
  command += k_bounded_generation_workload_id;
  command += " ";
  if (enable_internal) {
    command += "EMEL_BENCH_INTERNAL=1 ";
  }
  command += quote_arg_posix(bench_runner_binary_path().string());
  command += " --run-serialized-request ";
  command += quote_arg_posix(request_path.string());
  command += " --write-serialized-result ";
  command += quote_arg_posix(result_path.string());
  command += " > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());
#endif

  const int status = std::system(command.c_str());
  process_capture capture{};
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);
  result_text = read_file(result_path);

  std::error_code ec;
  std::filesystem::remove(request_path, ec);
  std::filesystem::remove(result_path, ec);
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
  command += "set EMEL_GENERATION_WORKLOAD_ID=";
  command += k_bounded_generation_workload_id;
  command += " && ";
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
  command += "EMEL_GENERATION_WORKLOAD_ID=";
  command += k_bounded_generation_workload_id;
  command += " ";
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

process_capture run_diarization_bench_capture(const std::string & mode,
                                              const bool emit_jsonl = false) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      (emit_jsonl ? ("diarization-jsonl-" + mode) : ("diarization-text-" + mode));
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path output_dir = tmp_dir / "outputs";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_BENCH_SUITE=diarization_sortformer && ";
  command += "set EMEL_BENCH_ITERS=1 && ";
  command += "set EMEL_BENCH_RUNS=1 && ";
  command += "set EMEL_BENCH_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_WARMUP_RUNS=0 && ";
  if (emit_jsonl) {
    command += "set EMEL_DIARIZATION_BENCH_FORMAT=jsonl && ";
    command += "set \"EMEL_DIARIZATION_RESULT_DIR=";
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
  command += "EMEL_BENCH_SUITE=diarization_sortformer ";
  command += "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_WARMUP_RUNS=0 ";
  if (emit_jsonl) {
    command += "EMEL_DIARIZATION_BENCH_FORMAT=jsonl ";
    command += "EMEL_DIARIZATION_RESULT_DIR=" + quote_arg_posix(output_dir.string()) + " ";
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

TEST_CASE("bench_runner generation compare keeps bounded maintained Liquid fixture") {
  const process_capture capture = run_generation_bench_compare_capture();
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("# generation_architecture: lfm2") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_formatter_contract:") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_stage_probe: case=" +
                                 std::string{k_bounded_generation_case_name}) !=
        std::string::npos);
  CHECK(capture.stdout_text.find("emel_prefill_linear_probe_ns=") != std::string::npos);
  CHECK(capture.stdout_text.find("reference_prefill_attention_probe_ns=") !=
        std::string::npos);

  CHECK(capture.stdout_text.find(k_bounded_generation_case_name) !=
        std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/"
                                 "lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_"
                                 "max_tokens_1000") == std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/"
                                 "qwen3_0_6b_q8_0_prompt_hello_max_tokens_1") ==
        std::string::npos);
  const std::string binary_size_line =
      find_line_with_prefix(capture.stdout_text, "# binary_size_compare:");
  CHECK_FALSE(binary_size_line.empty());
  CHECK(binary_size_line.find("status=ok") != std::string::npos);
  CHECK(parse_named_metric(binary_size_line, "emel_bytes") > 0u);
  CHECK(parse_named_metric(binary_size_line, "llama_bytes") > 0u);
}

TEST_CASE("bench_main delegates to runner-owned cli boundary") {
  const std::string main_source = read_file(repo_root() / "tools" / "bench" / "bench_main.cpp");
  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");

  CHECK(main_source.find("run_bench_cli(argc, argv)") != std::string::npos);
  CHECK(main_source.find("default_test_cases") == std::string::npos);
  CHECK(main_source.find("run_benchmarks") == std::string::npos);
  CHECK(main_source.find("print_compare") == std::string::npos);
  CHECK(runner_source.find("int emel::bench::run_bench_cli(int argc, char ** argv)") !=
        std::string::npos);
  CHECK(runner_source.find("EMEL_BENCH_ITERS") != std::string::npos);
  CHECK(runner_source.find("print_compare") != std::string::npos);
}

TEST_CASE("bench runner contract serializes requests and results for a process seam") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::compare;
  request.suite = "generation";
  request.cfg.iterations = 17u;
  request.cfg.runs = 3u;
  request.cfg.warmup_iterations = 5u;
  request.cfg.warmup_runs = 1u;
  request.generation_jsonl = true;

  const std::string serialized = emel::bench::serialize_runner_request(request);
  CHECK(serialized.find("schema=bench_runner_request/v1\n") != std::string::npos);
  CHECK(serialized.find("mode=compare\n") != std::string::npos);
  CHECK(serialized.find("suite=generation\n") != std::string::npos);

  emel::bench::runner_request parsed = {};
  CHECK(emel::bench::parse_runner_request(serialized, parsed));
  CHECK(parsed.mode == emel::bench::runner_mode::compare);
  CHECK(parsed.suite == "generation");
  CHECK(parsed.cfg.iterations == 17u);
  CHECK(parsed.cfg.runs == 3u);
  CHECK(parsed.cfg.warmup_iterations == 5u);
  CHECK(parsed.cfg.warmup_runs == 1u);
  CHECK(parsed.generation_jsonl);
  CHECK_FALSE(parsed.diarization_jsonl);

  emel::bench::runner_result result = {};
  result.exit_code = 2;
  result.error_kind = "invalid_request";
  result.error_message = "bad runner payload";
  const std::string result_text = emel::bench::serialize_runner_result(result);

  emel::bench::runner_result parsed_result = {};
  CHECK(emel::bench::parse_runner_result(result_text, parsed_result));
  CHECK(parsed_result.exit_code == 2);
  CHECK(parsed_result.error_kind == "invalid_request");
  CHECK(parsed_result.error_message == "bad runner payload");

  result.exit_code = -1;
  const std::string negative_result_text = emel::bench::serialize_runner_result(result);
  CHECK(negative_result_text.find("exit_code=-1\n") != std::string::npos);

  emel::bench::runner_result parsed_negative_result = {};
  CHECK(emel::bench::parse_runner_result(negative_result_text, parsed_negative_result));
  CHECK(parsed_negative_result.exit_code == -1);
  CHECK(parsed_negative_result.error_kind == "invalid_request");
  CHECK(parsed_negative_result.error_message == "bad runner payload");
}

TEST_CASE("benchmark snapshot value uses the median timing run") {
  const std::vector<double> sorted_samples{5.0, 8.0, 100.0};

  CHECK(emel::bench::select_reported_ns_per_op(sorted_samples) ==
        doctest::Approx(8.0));

  const std::vector<double> five_samples{5.0, 8.0, 9.0, 10.0, 100.0};

  CHECK(emel::bench::select_reported_ns_per_op(five_samples) ==
        doctest::Approx(9.0));
}

TEST_CASE("benchmark measurement clamps zero runs and iterations") {
  emel::bench::config cfg = {};
  std::uint32_t calls = 0;
  const auto measured = emel::bench::measure_case("bench/zero_cfg", cfg, [&]() {
    ++calls;
  });

  CHECK(calls == 1u);
  CHECK(measured.iterations == 1u);
  CHECK(measured.runs == 1u);
}

TEST_CASE("bench runner contract rejects malformed process payloads") {
  emel::bench::runner_request request = {};
  CHECK_FALSE(emel::bench::parse_runner_request("schema=bench_runner_request/v1\n", request));
  CHECK_FALSE(emel::bench::parse_runner_request(
    "schema=bench_runner_request/v1\n"
    "mode=unknown\n"
    "suite=generation\n"
    "iterations=1\n"
    "runs=1\n"
    "warmup_iterations=0\n"
    "warmup_runs=0\n"
    "generation_jsonl=0\n"
    "diarization_jsonl=0\n",
    request));
  CHECK_FALSE(emel::bench::parse_runner_request(
    "schema=bench_runner_request/v1\n"
    "mode=compare\n"
    "suite=generation\n"
    "iterations=one\n"
    "runs=1\n"
    "warmup_iterations=0\n"
    "warmup_runs=0\n"
    "generation_jsonl=0\n"
    "diarization_jsonl=0\n",
    request));

  emel::bench::runner_result result = {};
  CHECK_FALSE(emel::bench::parse_runner_result("schema=bench_runner_result/v1\n", result));
  CHECK_FALSE(emel::bench::parse_runner_result(
    "schema=bench_runner_result/v1\nexit_code=bad\n", result));
}

TEST_CASE("bench runner process seam executes a serialized request through the live binary") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::emel;
  request.suite = "generation";
  request.cfg.iterations = 1u;
  request.cfg.runs = 1u;
  request.cfg.warmup_iterations = 0u;
  request.cfg.warmup_runs = 0u;

  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(request),
                                     "process-seam-generation",
                                     result_text);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("# benchmark_config:") != std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/") != std::string::npos);

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 0);
  CHECK(result.error_kind.empty());
  CHECK(result.error_message.empty());
}

TEST_CASE("bench runner process seam writes deterministic errors for malformed payloads") {
  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture("schema=bench_runner_request/v1\n",
                                     "process-seam-malformed",
                                     result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("parse") != std::string::npos);
}

TEST_CASE("bench runner process seam writes deterministic errors for unknown modes") {
  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture(
        "schema=bench_runner_request/v1\n"
        "mode=unknown\n"
        "suite=generation\n"
        "iterations=1\n"
        "runs=1\n"
        "warmup_iterations=0\n"
        "warmup_runs=0\n"
        "generation_jsonl=0\n"
        "diarization_jsonl=0\n",
        "process-seam-unknown-mode",
        result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("parse") != std::string::npos);
}

TEST_CASE("bench runner process seam writes deterministic errors for unknown suites") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::emel;
  request.suite = "missing_suite";
  request.cfg.iterations = 1u;
  request.cfg.runs = 1u;
  request.cfg.warmup_iterations = 0u;
  request.cfg.warmup_runs = 0u;

  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(request),
                                     "process-seam-unknown-suite",
                                     result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "unknown_suite");
  CHECK_FALSE(result.error_message.empty());
}

TEST_CASE("bench runner process seam rejects conflicting jsonl output modes") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::emel;
  request.suite = "generation";
  request.cfg.iterations = 1u;
  request.cfg.runs = 1u;
  request.cfg.warmup_iterations = 0u;
  request.cfg.warmup_runs = 0u;
  request.generation_jsonl = true;
  request.diarization_jsonl = true;

  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(request),
                                     "process-seam-conflicting-jsonl",
                                     result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("jsonl") != std::string::npos);
}

TEST_CASE("bench runner process seam rejects incompatible jsonl suite requests") {
  emel::bench::runner_request generation_request = {};
  generation_request.mode = emel::bench::runner_mode::emel;
  generation_request.suite = "batch_planner";
  generation_request.cfg.iterations = 1u;
  generation_request.cfg.runs = 1u;
  generation_request.cfg.warmup_iterations = 0u;
  generation_request.cfg.warmup_runs = 0u;
  generation_request.generation_jsonl = true;

  std::string generation_result_text;
  const process_capture generation_capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(generation_request),
                                     "process-seam-bad-generation-jsonl-suite",
                                     generation_result_text);

  CHECK(generation_capture.exit_code == 2);
  CHECK(generation_capture.stdout_text.empty());

  emel::bench::runner_result generation_result = {};
  REQUIRE(emel::bench::parse_runner_result(generation_result_text, generation_result));
  CHECK(generation_result.exit_code == 2);
  CHECK(generation_result.error_kind == "invalid_request");
  CHECK(generation_result.error_message.find("generation jsonl") != std::string::npos);

  emel::bench::runner_request diarization_request = {};
  diarization_request.mode = emel::bench::runner_mode::reference;
  diarization_request.suite = "generation";
  diarization_request.cfg.iterations = 1u;
  diarization_request.cfg.runs = 1u;
  diarization_request.cfg.warmup_iterations = 0u;
  diarization_request.cfg.warmup_runs = 0u;
  diarization_request.diarization_jsonl = true;

  std::string diarization_result_text;
  const process_capture diarization_capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(diarization_request),
                                     "process-seam-bad-diarization-jsonl-suite",
                                     diarization_result_text);

  CHECK(diarization_capture.exit_code == 2);
  CHECK(diarization_capture.stdout_text.empty());

  emel::bench::runner_result diarization_result = {};
  REQUIRE(emel::bench::parse_runner_result(diarization_result_text, diarization_result));
  CHECK(diarization_result.exit_code == 2);
  CHECK(diarization_result.error_kind == "invalid_request");
  CHECK(diarization_result.error_message.find("diarization jsonl") != std::string::npos);
}

TEST_CASE("bench runner process seam rejects invalid serialized run counts") {
  emel::bench::runner_request zero_runs_request = {};
  zero_runs_request.mode = emel::bench::runner_mode::emel;
  zero_runs_request.suite = "batch_planner";
  zero_runs_request.cfg.iterations = 1u;
  zero_runs_request.cfg.runs = 0u;
  zero_runs_request.cfg.warmup_iterations = 0u;
  zero_runs_request.cfg.warmup_runs = 0u;

  std::string zero_runs_result_text;
  const process_capture zero_runs_capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(zero_runs_request),
                                     "process-seam-zero-runs",
                                     zero_runs_result_text);

  CHECK(zero_runs_capture.exit_code == 2);
  CHECK(zero_runs_capture.stdout_text.empty());

  emel::bench::runner_result zero_runs_result = {};
  REQUIRE(emel::bench::parse_runner_result(zero_runs_result_text, zero_runs_result));
  CHECK(zero_runs_result.exit_code == 2);
  CHECK(zero_runs_result.error_kind == "invalid_request");
  CHECK(zero_runs_result.error_message.find("runs") != std::string::npos);

  emel::bench::runner_request too_many_runs_request = zero_runs_request;
  too_many_runs_request.cfg.runs = 26u;

  std::string too_many_runs_result_text;
  const process_capture too_many_runs_capture =
      run_serialized_request_capture(emel::bench::serialize_runner_request(too_many_runs_request),
                                     "process-seam-too-many-runs",
                                     too_many_runs_result_text);

  CHECK(too_many_runs_capture.exit_code == 2);
  CHECK(too_many_runs_capture.stdout_text.empty());

  emel::bench::runner_result too_many_runs_result = {};
  REQUIRE(emel::bench::parse_runner_result(too_many_runs_result_text, too_many_runs_result));
  CHECK(too_many_runs_result.exit_code == 2);
  CHECK(too_many_runs_result.error_kind == "invalid_request");
  CHECK(too_many_runs_result.error_message.find("runs") != std::string::npos);

  emel::bench::runner_request too_many_warmups_request = zero_runs_request;
  too_many_warmups_request.cfg.runs = 1u;
  too_many_warmups_request.cfg.warmup_runs = 26u;

  std::string too_many_warmups_result_text;
  const process_capture too_many_warmups_capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(too_many_warmups_request),
      "process-seam-too-many-warmup-runs",
      too_many_warmups_result_text);

  CHECK(too_many_warmups_capture.exit_code == 2);
  CHECK(too_many_warmups_capture.stdout_text.empty());

  emel::bench::runner_result too_many_warmups_result = {};
  REQUIRE(emel::bench::parse_runner_result(too_many_warmups_result_text,
                                           too_many_warmups_result));
  CHECK(too_many_warmups_result.exit_code == 2);
  CHECK(too_many_warmups_result.error_kind == "invalid_request");
  CHECK(too_many_warmups_result.error_message.find("warmup_runs") != std::string::npos);
}

TEST_CASE("benchmark runner registration is localized outside the orchestrator") {
  CHECK(emel::bench::registered_runner_count() >= 29u);
  CHECK(emel::bench::find_registered_runner("generation") != nullptr);
  CHECK(emel::bench::find_registered_runner("diarization_sortformer") != nullptr);
  CHECK(emel::bench::find_registered_runner("tokenizer") != nullptr);
  CHECK(emel::bench::find_registered_runner("missing_suite") == nullptr);

  bool saw_generation = false;
  bool saw_tokenizer = false;
  for (std::size_t i = 0; i < emel::bench::registered_runner_count(); ++i) {
    saw_generation = saw_generation ||
      emel::bench::registered_runner_suite_at(i) == std::string_view{"generation"};
    saw_tokenizer = saw_tokenizer ||
      emel::bench::registered_runner_suite_at(i) == std::string_view{"tokenizer"};
  }
  CHECK(saw_generation);
  CHECK(saw_tokenizer);
}

TEST_CASE("bench runner orchestration no longer owns broad static registration") {
  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  const std::string registry_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner_registry.cpp");

  CHECK(runner_source.find("std::array<bench::test_case") == std::string::npos);
  CHECK(runner_source.find("append_emel_generation_cases") == std::string::npos);
  CHECK(runner_source.find("append_reference_generation_cases") == std::string::npos);
  CHECK(runner_source.find("bench::default_runner_cases()") != std::string::npos);
  CHECK(registry_source.find("append_emel_generation_cases") != std::string::npos);
  CHECK(registry_source.find("append_reference_generation_cases") != std::string::npos);
}

TEST_CASE("bench runner suites build through independent object targets") {
  const std::string cmake_source = read_file(repo_root() / "tools" / "bench" / "CMakeLists.txt");

  CHECK(cmake_source.find("bench_runner_suite_${suite_name}") != std::string::npos);
  CHECK(cmake_source.find("add_library(${target_name} OBJECT") != std::string::npos);
  CHECK(cmake_source.find("$<TARGET_OBJECTS:${target_name}>") != std::string::npos);
  CHECK(cmake_source.find("configure_bench_runner_common_target(${target_name})") !=
        std::string::npos);
  CHECK(cmake_source.find("configure_bench_runner_artifact_definitions(${target_name})") !=
        std::string::npos);
  CHECK(cmake_source.find("add_bench_runner_suite(generation generation_bench.cpp") !=
        std::string::npos);
  CHECK(cmake_source.find("add_bench_runner_suite(diarization_sortformer") != std::string::npos);
  CHECK(cmake_source.find("EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_kv\"") !=
        std::string::npos);
  CHECK(cmake_source.find("EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_recurrent\"") !=
        std::string::npos);
  CHECK(cmake_source.find("EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_hybrid\"") !=
        std::string::npos);
  CHECK(cmake_source.find("BENCH_RUNNER_SUITE_TARGETS") != std::string::npos);
}

TEST_CASE("benchmark dependency manifest covers registered runners conservatively") {
  namespace manifest = emel::bench::dependency_manifest;

  CHECK(manifest::kind_name(manifest::dependency_kind::source) == "source");
  CHECK(manifest::kind_name(manifest::dependency_kind::config) == "config");
  CHECK_FALSE(manifest::requires_full_gate({}));
  CHECK(manifest::requires_full_gate({.missing = true}));
  CHECK(manifest::requires_full_gate({.stale = true}));
  CHECK(manifest::requires_full_gate({.uncertain = true}));

  const auto all_records = manifest::records_for("all");
  REQUIRE_FALSE(all_records.empty());
  const std::size_t total_all_records = static_cast<std::size_t>(std::count_if(
      manifest::records().begin(),
      manifest::records().end(),
      [](const auto & record) { return record.runner == std::string_view{"all"}; }));
  CHECK(all_records.size() == total_all_records);

  bool saw_cmake = false;
  bool saw_quality_gate = false;
  for (const auto & record : all_records) {
    saw_cmake = saw_cmake || record.path == std::string_view{"tools/bench/CMakeLists.txt"};
    saw_quality_gate = saw_quality_gate ||
      record.path == std::string_view{"scripts/quality_gates.sh"};
  }
  CHECK(saw_cmake);
  CHECK(saw_quality_gate);

  for (std::size_t i = 0; i < emel::bench::registered_runner_count(); ++i) {
    const std::string_view runner = emel::bench::registered_runner_suite_at(i);
    const auto records = manifest::records_for(runner);
    CHECK_MESSAGE(!records.empty(), "missing manifest records for runner " << runner);
    const std::size_t total_records = static_cast<std::size_t>(std::count_if(
        manifest::records().begin(),
        manifest::records().end(),
        [runner](const auto & record) { return record.runner == runner; }));
    CHECK_MESSAGE(records.size() == total_records,
                  "manifest records are not contiguous for runner " << runner);
    bool has_source = false;
    for (const auto & record : records) {
      has_source = has_source || record.kind == manifest::dependency_kind::source;
    }
    CHECK_MESSAGE(has_source, "runner lacks source record " << runner);
  }

  const auto generation_records = manifest::records_for("generation");
  bool has_generation_config = false;
  bool has_generation_model = false;
  bool has_generation_script = false;
  for (const auto & record : generation_records) {
    has_generation_config =
      has_generation_config || record.kind == manifest::dependency_kind::config;
    has_generation_model =
      has_generation_model || record.kind == manifest::dependency_kind::model;
    has_generation_script =
      has_generation_script || record.kind == manifest::dependency_kind::script;
  }
  CHECK(has_generation_config);
  CHECK(has_generation_model);
  CHECK(has_generation_script);
  CHECK(manifest::records_for("missing_runner").empty());
}

TEST_CASE("benchmark dependency manifest renders and writes deterministic output") {
  namespace manifest = emel::bench::dependency_manifest;

  const std::string rendered = manifest::render();
  CHECK(rendered == manifest::render());
  CHECK(rendered.rfind(std::string(manifest::k_schema) + "\n", 0u) == 0u);
  CHECK(rendered.find("full_gate_on=missing,stale,uncertain\n") != std::string::npos);
  CHECK(rendered.find(
          "record runner=generation kind=source path=tools/bench/generation_bench.cpp") !=
        std::string::npos);
  CHECK(rendered.find(
          "record runner=diarization_sortformer kind=source "
          "path=tools/bench/diarization/sortformer_bench.cpp") != std::string::npos);

  const std::filesystem::path manifest_path =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "bench-dependency-manifest.txt";
  std::filesystem::create_directories(manifest_path.parent_path());
  REQUIRE(manifest::write(manifest_path));
  CHECK(read_file(manifest_path) == rendered);
  std::filesystem::remove(manifest_path);

  const std::string baseline =
      read_file(repo_root() / "tools" / "bench" / "dependency_manifest.txt");
  CHECK(baseline == rendered);
  const std::string docs =
      read_file(repo_root() / "tools" / "bench" / "dependency_manifest.md");
  CHECK(docs.find(manifest::k_schema) != std::string::npos);
  CHECK(docs.find("full_gate_on=missing,stale,uncertain") != std::string::npos);
}

TEST_CASE("bench_runner cli emits and checks dependency manifest freshness") {
  namespace manifest = emel::bench::dependency_manifest;

  const std::filesystem::path manifest_path =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "bench-runner-cli-dependency-manifest.txt";
  std::filesystem::create_directories(manifest_path.parent_path());

  process_capture write_capture = run_bench_runner_capture(
    {"--write-dependency-manifest", manifest_path.string()},
    "bench-manifest-write");
  CHECK(write_capture.exit_code == 0);
  CHECK(write_capture.stderr_text.empty());
  CHECK(write_capture.stdout_text.find("dependency_manifest: action=write") !=
        std::string::npos);
  CHECK(write_capture.stdout_text.find("schema=bench_dependency_manifest/v1") !=
        std::string::npos);
  CHECK(read_file(manifest_path) == manifest::render());

  process_capture fresh_capture = run_bench_runner_capture(
    {"--check-dependency-manifest", manifest_path.string()},
    "bench-manifest-fresh");
  CHECK(fresh_capture.exit_code == 0);
  CHECK(fresh_capture.stdout_text.find("full_gate=0") != std::string::npos);
  CHECK(fresh_capture.stdout_text.find("reason=fresh") != std::string::npos);

  process_capture uncertain_capture = run_bench_runner_capture(
    {"--check-dependency-manifest", manifest_path.string(), "--dependency-manifest-uncertain"},
    "bench-manifest-uncertain");
  CHECK(uncertain_capture.exit_code == 3);
  CHECK(uncertain_capture.stdout_text.find("full_gate=1") != std::string::npos);
  CHECK(uncertain_capture.stdout_text.find("reason=uncertain") != std::string::npos);

  {
    std::ofstream stale_manifest(manifest_path, std::ios::binary);
    stale_manifest << "stale manifest\n";
  }
  process_capture stale_capture = run_bench_runner_capture(
    {"--check-dependency-manifest", manifest_path.string()},
    "bench-manifest-stale");
  CHECK(stale_capture.exit_code == 3);
  CHECK(stale_capture.stdout_text.find("reason=stale") != std::string::npos);

  std::filesystem::remove(manifest_path);
  process_capture missing_capture = run_bench_runner_capture(
    {"--check-dependency-manifest", manifest_path.string()},
    "bench-manifest-missing");
  CHECK(missing_capture.exit_code == 3);
  CHECK(missing_capture.stdout_text.find("reason=missing") != std::string::npos);

  process_capture invalid_capture = run_bench_runner_capture(
    {"--write-dependency-manifest", manifest_path.string(), "--dependency-manifest-uncertain"},
    "bench-manifest-invalid");
  CHECK(invalid_capture.exit_code == 2);
  CHECK(invalid_capture.stderr_text.find("error: invalid dependency manifest arguments") !=
        std::string::npos);
}

TEST_CASE("shared benchmark orchestration stays lane-neutral and actor-boundary clean") {
  const std::vector<std::filesystem::path> shared_paths = {
    repo_root() / "tools" / "bench" / "bench_main.cpp",
    repo_root() / "tools" / "bench" / "bench_runner.cpp",
    repo_root() / "tools" / "bench" / "bench_runner.hpp",
    repo_root() / "tools" / "bench" / "bench_runner_contract.hpp",
    repo_root() / "tools" / "bench" / "bench_runner_registry.cpp",
    repo_root() / "tools" / "bench" / "bench_runner_registry.hpp",
    repo_root() / "tools" / "bench" / "bench_dependency_manifest.cpp",
    repo_root() / "tools" / "bench" / "bench_dependency_manifest.hpp",
  };
  const std::vector<std::string> forbidden_patterns = {
    "/actions.hpp",
    "/guards.hpp",
    "/detail.hpp",
    "emel::text::generator::action::",
    "emel::text::generator::guard::",
    "emel::text::generator::detail::",
    "llama_model",
    "llama_context",
    "llama_vocab",
    "ggml_context",
    "shared_runtime",
    "shared_cache",
  };

  for (const std::filesystem::path & path : shared_paths) {
    const std::string source = read_file(path);
    REQUIRE_MESSAGE(!source.empty(), "missing source " << path.string());
    for (const std::string & pattern : forbidden_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.string() << " contains forbidden pattern " << pattern);
    }
  }

  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  CHECK(runner_source.find("append_emel_generation_cases") == std::string::npos);
  CHECK(runner_source.find("append_reference_generation_cases") == std::string::npos);
  CHECK(runner_source.find("append_emel_sortformer_diarization_cases") == std::string::npos);
  CHECK(runner_source.find("append_reference_sortformer_diarization_cases") == std::string::npos);
}

TEST_CASE("maintained benchmark runner sources avoid actor internals") {
  const std::vector<std::string> forbidden_patterns = {
    "/actions.hpp",
    "/guards.hpp",
    "::action::",
    "::guard::",
    "emel/batch/planner/detail.hpp",
    "emel/diarization/request/detail.hpp",
    "emel/diarization/sortformer/executor/detail.hpp",
    "emel/diarization/sortformer/pipeline/detail.hpp",
    "emel/text/generator/detail.hpp",
    "emel/text/generator/prefill/detail.hpp",
    "emel/text/jinja/formatter/detail.hpp",
    "emel/text/jinja/parser/detail.hpp",
    "emel::batch::planner::detail::",
    "emel::diarization::request::detail::",
    "emel::diarization::sortformer::executor::detail::",
    "emel::diarization::sortformer::pipeline::detail::",
    "emel::text::generator::detail::",
    "emel::text::generator::prefill::detail::",
    "emel::text::jinja::formatter::detail::",
    "emel::text::jinja::parser::detail::",
  };

  std::size_t checked_files = 0u;
  const std::filesystem::path bench_dir = repo_root() / "tools" / "bench";
  for (const auto & entry : std::filesystem::recursive_directory_iterator(bench_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::filesystem::path path = entry.path();
    const std::string ext = path.extension().string();
    if (ext != ".cpp" && ext != ".hpp") {
      continue;
    }
    if (path.filename() == "bench_runner_tests.cpp") {
      continue;
    }

    const std::string source = read_file(path);
    REQUIRE_MESSAGE(!source.empty(), "missing source " << path.string());
    checked_files += 1u;
    for (const std::string & pattern : forbidden_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.string() << " contains forbidden actor-internal pattern " << pattern);
    }
  }
  CHECK(checked_files > 20u);
}

TEST_CASE("maintained benchmark behavior coverage remains source-backed") {
  const std::string tests_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner_tests.cpp");

  CHECK(tests_source.find("bench_main shim delegates to benchmark runner cli") !=
        std::string::npos);
  CHECK(tests_source.find("bench_runner generation jsonl emits manifest-driven workload metadata") !=
        std::string::npos);
  CHECK(tests_source.find("bench_runner diarization jsonl emits structured maintained parity metadata") !=
        std::string::npos);
  CHECK(tests_source.find("benchmark runner registration is localized outside the orchestrator") !=
        std::string::npos);
  CHECK(tests_source.find("benchmark dependency manifest renders and writes deterministic output") !=
        std::string::npos);
  CHECK(tests_source.find("shared benchmark orchestration stays lane-neutral") !=
        std::string::npos);
  CHECK(tests_source.find("maintained benchmark runner sources avoid actor internals") !=
        std::string::npos);
}

TEST_CASE("generation_stage_probe_emel_path_does_not_bypass_generator_actor") {
  const std::string source =
      read_file(repo_root() / "tools" / "bench" / "generation_bench.cpp");
  REQUIRE_FALSE(source.empty());

  const std::vector<std::string> forbidden_source_patterns = {
    "#include \"emel/text/generator/detail.hpp\"",
    "#include \"emel/text/generator/actions.hpp\"",
    "#include \"emel/text/generator/guards.hpp\"",
    "#include \"emel/text/generator/prefill/guards.hpp\"",
    "emel::text::generator::detail::",
    "emel::text::generator::action::",
    "emel::text::generator::guard::",
    "emel::text::generator::prefill::guard::",
    "->generation_",
  };
  for (const std::string & pattern : forbidden_source_patterns) {
    CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                  "generation_bench.cpp contains forbidden pattern " << pattern);
  }

  const std::string marker = "bool measure_emel_stage_probe(";
  const auto start = source.find(marker);
  REQUIRE(start != std::string::npos);
  const auto end = source.find("bool measure_reference_stage_probe(", start);
  REQUIRE(end != std::string::npos);

  const std::string probe_source = source.substr(start, end - start);
  CHECK(probe_source.find("emel::text::generator::detail::") == std::string::npos);
  CHECK(probe_source.find("emel::text::generator::guard::") == std::string::npos);
  CHECK(probe_source.find("emel::text::generator::prefill::guard::") == std::string::npos);
  CHECK(probe_source.find("emel::text::generator::action::context") == std::string::npos);
  CHECK(probe_source.find("emel::text::generator::prefill::action::context") ==
        std::string::npos);
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
  CHECK(emel_capture.stdout_text.find("\"workload_id\":\"" +
                                      std::string{k_bounded_generation_workload_id} +
                                      "\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_manifest_path\":\"tools/bench/generation_variants/"
            "lfm2/single_user_hello/parity/max_tokens_1.json\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"prompt_fixture_id\":\"single_user_hello_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"prompt_fixture_path\":\"tools/bench/generation_prompts/single_user_hello.json\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"prompt_id\":\"single_user:hello\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"formatter_mode\":\"chat_template_supported_v1\"") !=
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
  CHECK(reference_capture.stdout_text.find("\"workload_manifest_path\":\"tools/bench/generation_variants/") !=
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

TEST_CASE("bench_runner diarization jsonl emits structured maintained parity metadata") {
  const process_capture emel_capture = run_diarization_bench_capture("emel", true);
  CHECK(emel_capture.exit_code == 0);
  CHECK(emel_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"schema\":\"diarization_compare/v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"emel\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"reference\"") == std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"backend_id\":\"emel.diarization.sortformer\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparison_mode\":\"parity\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"model_id\":\"diar_streaming_sortformer_4spk_v2_1_gguf\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"fixture_id\":\"ami_en2002b_mix_headset_137.00_152.04_16khz_mono\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_id\":\"diarization_sortformer_pipeline_v1\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparable\":true") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"output_path\":\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("ns/op,") == std::string::npos);

  const process_capture reference_capture = run_diarization_bench_capture("reference", true);
  CHECK(reference_capture.exit_code == 0);
  CHECK(reference_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"schema\":\"diarization_compare/v1\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"reference\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"emel\"") == std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"backend_id\":\"recorded.diarization.baseline\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"comparison_mode\":\"parity\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"comparable\":true") != std::string::npos);
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

TEST_CASE("generation workload manifests are discovered deterministically") {
  std::vector<emel::bench::generation_workload_manifest> manifests = {};
  std::string error = {};
  CHECK(emel::bench::load_generation_workload_manifests(repo_root(), manifests, &error));
  CHECK(error.empty());
  CHECK(manifests.size() >= 13u);
  REQUIRE(!manifests.empty());
  CHECK(manifests.front().workload_manifest_path.find("tools/bench/generation_variants/") == 0u);
  CHECK(manifests.front().workload_manifest_path !=
        "tools/bench/generation_variants/" +
          std::filesystem::path(manifests.front().workload_manifest_path).filename().string());
  CHECK(std::any_of(manifests.begin(), manifests.end(), [](const auto & manifest) {
    return manifest.id == "qwen3_single_user_hello_max_tokens_1_v1";
  }));
}
