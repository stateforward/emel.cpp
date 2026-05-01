#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "../generation_fixture_registry.hpp"
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
  CHECK(emel_capture.stdout_text.find("\"workload_id\":\"qwen3_single_user_hello_max_tokens_1_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_manifest_path\":\"tools/bench/generation_variants/"
            "qwen3/single_user_hello/max_tokens_1.json\"") != std::string::npos);
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
