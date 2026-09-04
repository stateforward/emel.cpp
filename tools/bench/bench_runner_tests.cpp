#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <array>
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
#include "model/needle/request_aggregation.hpp"
#include "model/needle/request_fixture_contract.hpp"

#if defined(_WIN32)
#include <process.h>
#endif
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
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

std::filesystem::path bench_moshi_lm_compare_wrapper_path() {
  return repo_root() / "scripts" / "bench_moshi_lm_compare.sh";
}

std::filesystem::path cactus_reference_driver_path() {
  return repo_root() / "tools" / "bench" / "model" / "needle" /
         "cactus_reference.py";
}
std::filesystem::path needle_authenticated_exec_test_path() {
#ifdef BENCH_NEEDLE_AUTHENTICATED_EXEC_TEST_PATH
  return BENCH_NEEDLE_AUTHENTICATED_EXEC_TEST_PATH;
#else
  return std::filesystem::path("needle_authenticated_exec_tests");
#endif
}
std::string needle_clean_environment_prefix() {
  return "env -u LD_PRELOAD -u LD_LIBRARY_PATH -u LD_AUDIT "
         "-u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES "
         "-u DYLD_FRAMEWORK_PATH -u DYLD_FALLBACK_LIBRARY_PATH "
         "-u DYLD_FALLBACK_FRAMEWORK_PATH -u PYTHONPATH -u PYTHONHOME "
         "-u PYTHONSTARTUP -u PYTHONINSPECT ";
}


constexpr const char *k_bounded_generation_workload_id =
    "lfm2_single_user_hello_max_tokens_1_v1";
constexpr const char *k_bounded_generation_case_name =
    "generation/preloaded_request/"
    "lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1";

std::filesystem::path maintained_generation_fixture_path(
    const emel::tools::generation_fixture_registry::maintained_fixture
        &fixture) {
  return repo_root() / fixture.fixture_rel;
}

bool maintained_generation_fixture_exists(
    const emel::tools::generation_fixture_registry::maintained_fixture
        &fixture) {
  return std::filesystem::exists(maintained_generation_fixture_path(fixture));
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input},
                     std::istreambuf_iterator<char>{}};
}

void write_file(const std::filesystem::path &path,
                const std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  REQUIRE(output);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  REQUIRE(output);
}

void make_executable(const std::filesystem::path &path) {
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::add);
}

void replace_all(std::string &text, const std::string_view needle,
                 const std::string_view replacement) {
  std::size_t pos = 0u;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
}

std::string actor_boundary_scan_source(const std::filesystem::path &path,
                                       std::string source) {
  const std::string generic_path = path.generic_string();
  if (generic_path.find("tools/bench/bench_dependency_manifest.cpp") !=
      std::string::npos) {
    replace_all(source, "src/emel/text/generator/detail.hpp",
                "src/emel/text/generator/lane_contract");
  }
  if (generic_path.find("tools/bench/kernel/x86_64_bench.cpp") !=
      std::string::npos) {
    replace_all(source, "emel::kernel::x86_64::action::context",
                "emel::kernel::x86_64::context");
  }
  return source;
}

std::string quote_arg_posix(const std::string &arg) {
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

std::string quote_arg_windows(const std::string &arg) {
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

process_capture run_command_capture(const std::string &command,
                                    const std::filesystem::path &stdout_path,
                                    const std::filesystem::path &stderr_path) {
  const int status = std::system(command.c_str());
  process_capture capture{};
  capture.stdout_text = read_file(stdout_path);
  capture.stderr_text = read_file(stderr_path);

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

std::uint64_t current_process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(::_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

process_capture run_bench_runner_capture(const std::vector<std::string> &args,
                                         const std::string &tag) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" / tag;
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  std::string command;
#if defined(_WIN32)
  command = quote_arg_windows(bench_runner_binary_path().string());
  for (const std::string &arg : args) {
    command += " " + quote_arg_windows(arg);
  }
  command += " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += quote_arg_posix(bench_runner_binary_path().string());
  for (const std::string &arg : args) {
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

process_capture
run_serialized_request_capture(const std::string_view request_text,
                               const std::string &tag, std::string &result_text,
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

process_capture run_generation_bench_capture(const std::string &mode,
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
    command +=
        "EMEL_GENERATION_RESULT_DIR=" + quote_arg_posix(output_dir.string()) +
        " ";
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

process_capture run_suite_bench_capture(const std::string &suite,
                                        const std::string &mode,
                                        const std::string &tag,
                                        const bool enable_internal = false) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" / tag;
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_BENCH_SUITE=" + suite + " && ";
  command += "set EMEL_BENCH_ITERS=1 && ";
  command += "set EMEL_BENCH_RUNS=1 && ";
  command += "set EMEL_BENCH_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_WARMUP_RUNS=0 && ";
  if (enable_internal) {
    command += "set EMEL_BENCH_INTERNAL=1 && ";
  }
  command += quote_arg_windows(bench_runner_binary_path().string());
  command += " --mode=" + mode + " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += "EMEL_BENCH_SUITE=" + quote_arg_posix(suite) + " ";
  command += "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_WARMUP_RUNS=0 ";
  if (enable_internal) {
    command += "EMEL_BENCH_INTERNAL=1 ";
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

std::size_t count_benchmark_rows(const std::string_view output) {
  std::size_t count = 0u;
  std::size_t cursor = 0u;
  while (cursor < output.size()) {
    const std::size_t end = output.find('\n', cursor);
    const std::string_view line =
        end == std::string_view::npos
            ? output.substr(cursor)
            : output.substr(cursor, end - cursor);
    if (!line.empty() && line.front() != '#') {
      ++count;
    }
    cursor = end == std::string_view::npos ? output.size() : end + 1u;
  }
  return count;
}

process_capture run_needle_graph_bench_capture(const std::string &mode,
                                               const std::string &tag) {
  static std::uint64_t invocation = 0u;
  const std::filesystem::path tmp_root =
      std::filesystem::temp_directory_path() /
      ("emel-bench-runner-tests-" + std::to_string(current_process_id()));
  std::error_code ec;
  REQUIRE(std::filesystem::create_directory(tmp_root, ec));
  REQUIRE_FALSE(ec);
#if !defined(_WIN32)
  std::filesystem::permissions(
      tmp_root, std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace, ec);
  REQUIRE_FALSE(ec);
#endif
  const std::filesystem::path tmp_dir =
      tmp_root / (tag + "-" + std::to_string(++invocation));
  std::filesystem::remove_all(tmp_dir, ec);
  REQUIRE_FALSE(ec);
  REQUIRE(std::filesystem::create_directories(tmp_dir));
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";

  std::string command;
#if defined(_WIN32)
  command = "set EMEL_BENCH_NEEDLE_REQUEST_COMPARE=1 && ";
  command += "set EMEL_BENCH_SUITE=needle_graph && ";
  command += "set EMEL_BENCH_ITERS=1 && ";
  command += "set EMEL_BENCH_RUNS=1 && ";
  command += "set EMEL_BENCH_WARMUP_ITERS=0 && ";
  command += "set EMEL_BENCH_WARMUP_RUNS=0 && ";
  command += "set EMEL_BENCH_NEEDLE_GRAPH_DECODE_ITERS=1 && ";
  command += "set EMEL_BENCH_NEEDLE_GRAPH_PREFILL_ITERS=1 && ";
  command += "set EMEL_BENCH_NEEDLE_HADAMARD_ITERS=1 && ";
  command += "set EMEL_BENCH_NEEDLE_FWHT_ITERS=1 && ";
  command += "set EMEL_BENCH_NEEDLE_SWA_ITERS=1 && ";
  command += quote_arg_windows(bench_runner_binary_path().string());
  command += " --mode=" + mode + " > ";
  command += quote_arg_windows(stdout_path.string());
  command += " 2> ";
  command += quote_arg_windows(stderr_path.string());
#else
  command = "ulimit -s 8192; ";
  command += "EMEL_BENCH_NEEDLE_REQUEST_COMPARE=1 ";
  command += "EMEL_BENCH_SUITE=needle_graph ";
  command += "EMEL_BENCH_ITERS=1 ";
  command += "EMEL_BENCH_RUNS=1 ";
  command += "EMEL_BENCH_WARMUP_ITERS=0 ";
  command += "EMEL_BENCH_WARMUP_RUNS=0 ";
  command += "EMEL_BENCH_NEEDLE_GRAPH_DECODE_ITERS=1 ";
  command += "EMEL_BENCH_NEEDLE_GRAPH_PREFILL_ITERS=1 ";
  command += "EMEL_BENCH_NEEDLE_HADAMARD_ITERS=1 ";
  command += "EMEL_BENCH_NEEDLE_FWHT_ITERS=1 ";
  command += "EMEL_BENCH_NEEDLE_SWA_ITERS=1 ";
  command += quote_arg_posix(bench_runner_binary_path().string());
  command += " --mode=" + mode + " > ";
  command += quote_arg_posix(stdout_path.string());
  command += " 2> ";
  command += quote_arg_posix(stderr_path.string());
#endif

  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  std::filesystem::remove_all(tmp_dir, ec);
  CHECK_FALSE(ec);
  std::filesystem::remove(tmp_root, ec);
  CHECK_FALSE(ec);
  return capture;
}

process_capture run_diarization_bench_capture(const std::string &mode,
                                              const bool emit_jsonl = false) {
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      (emit_jsonl ? ("diarization-jsonl-" + mode)
                  : ("diarization-text-" + mode));
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
    command +=
        "EMEL_DIARIZATION_RESULT_DIR=" + quote_arg_posix(output_dir.string()) +
        " ";
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

std::uint64_t parse_named_metric(const std::string &haystack,
                                 const std::string &name) {
  const std::string needle = name + "=";
  const size_t pos = haystack.find(needle);
  if (pos == std::string::npos) {
    return 0u;
  }

  size_t cursor = pos + needle.size();
  std::uint64_t value = 0u;
  while (cursor < haystack.size() && haystack[cursor] >= '0' &&
         haystack[cursor] <= '9') {
    value = value * 10u + static_cast<std::uint64_t>(haystack[cursor] - '0');
    ++cursor;
  }
  return value;
}
double parse_named_double(const std::string &haystack, const std::string &name) {
  const std::string needle = name + "=";
  const size_t pos = haystack.find(needle);
  if (pos == std::string::npos) {
    return 0.0;
  }
  return std::stod(haystack.substr(pos + needle.size()));
}

std::string find_line_with_prefix(const std::string &haystack,
                                  const std::string &prefix) {
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
} // namespace

TEST_CASE("needle request aggregation independently ranks phase metrics") {
  using emel::bench::needle_request::aggregate_runs;
  using emel::bench::needle_request::run_sample;
  const std::vector<run_sample> samples = {
      {.wall_ns = 10.0, .prefill_ns = 300.0, .decode_ns = 2000.0,
       .prompt_tokens = 100u, .decode_tokens = 20u, .envelopes = {"same"}},
      {.wall_ns = 20.0, .prefill_ns = 100.0, .decode_ns = 3000.0,
       .prompt_tokens = 100u, .decode_tokens = 20u, .envelopes = {"same"}},
      {.wall_ns = 30.0, .prefill_ns = 200.0, .decode_ns = 1000.0,
       .prompt_tokens = 100u, .decode_tokens = 20u, .envelopes = {"same"}},
  };
  run_sample aggregated;
  REQUIRE(aggregate_runs(samples, aggregated));
  CHECK(aggregated.wall_ns == doctest::Approx(20.0));
  CHECK(aggregated.prefill_ns == doctest::Approx(200.0));
  CHECK(aggregated.decode_ns == doctest::Approx(2000.0));
  CHECK(aggregated.prompt_tokens == 100u);
  CHECK(aggregated.decode_tokens == 20u);
  CHECK(aggregated.envelopes == std::vector<std::string>{"same"});

  std::vector<run_sample> unstable = samples;
  unstable.back().envelopes = {"different"};
  CHECK_FALSE(aggregate_runs(unstable, aggregated));
  unstable = samples;
  unstable.back().prompt_tokens += 1u;
  CHECK_FALSE(aggregate_runs(unstable, aggregated));
  unstable = samples;
  unstable.back().decode_tokens += 1u;
  CHECK_FALSE(aggregate_runs(unstable, aggregated));
}

TEST_CASE("needle request fixture rejects altered token ids") {
  using emel::bench::needle_request::token_ids_match;
  const std::array<int32_t, 4> canonical = {1, 2, 3, 4};
  std::array<int32_t, 4> actual = canonical;
  CHECK(token_ids_match(canonical, actual));
  actual[2] = 99;
  CHECK_FALSE(token_ids_match(canonical, actual));
  const std::array<int32_t, 3> truncated = {1, 2, 3};
  CHECK_FALSE(token_ids_match(canonical, truncated));
}

TEST_CASE("needle cactus boundary rejects substituted inputs and invalid values") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import contextlib
import importlib.util
import json
import io
import math
import os
import pathlib
import sys
import tempfile
import types

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def rejected(call):
    try:
        call()
    except SystemExit:
        return
    raise AssertionError("boundary accepted invalid input")

root = pathlib.Path(tempfile.mkdtemp())
substitute = root / "substitute.cact"
substitute.write_bytes(b"not the canonical model")
rejected(lambda: module.validate_canonical_input(
    substitute, module.MODEL_SHA256, "model"))
rejected(lambda: module.validate_canonical_path(
    substitute, pathlib.Path(sys.argv[1]).resolve().parents[3] /
    module.MODEL_RELATIVE_PATH, "model"))
rejected(lambda: module.exact_int(True, "runs", minimum=1))
rejected(lambda: module.exact_int(33, "runs", minimum=1))
for value in (True, 0, -1, math.nan, math.inf, -math.inf, "1"):
    rejected(lambda value=value: module.positive_finite_number(value, "metric"))
fake_package = root / "needle"
fake_package.mkdir()
fake_init = fake_package / "__init__.py"
fake_init.write_text("__version__ = '2.0.8'\n")
rejected(lambda: module.validate_needle_package(root))

fake_module = types.SimpleNamespace(
    __version__="substitute", __file__=str(fake_init), Needle=object,
    _library_path=lambda: root / "libneedle.so")
rejected(lambda: module.validate_needle_module_identity(fake_module, fake_package))
fake_module.__version__ = module.NEEDLE_PACKAGE_VERSION
fake_module.__file__ = str(root / "other.py")
rejected(lambda: module.validate_needle_module_identity(fake_module, fake_package))

fake_library = root / "libneedle.so"
fake_library.write_bytes(b"not the canonical native runtime")
fake_module.__file__ = str(fake_init)
rejected(lambda: module.validate_needle_native_library(fake_module))
os.environ["NEEDLE_LIB_PATH"] = str(fake_library)
rejected(lambda: module.validate_needle_native_library(fake_module))
del os.environ["NEEDLE_LIB_PATH"]
reference = {
    "schema": module.SCHEMA,
    "lane": "reference",
    "backend_id": "cactus.libneedle.native",
    "backend_language": "python_ctypes_native",
    "reference_source": "live",
    "model_id": module.MODEL_ID,
    "model_path": module.MODEL_RELATIVE_PATH,
    "fixture_id": module.FIXTURE_ID,
    "workload_id": module.WORKLOAD_ID,
    "thread_count": 1,
    "thread_contract": module.THREAD_CONTRACT,
    "prompt_rows": module.PROMPT_ROWS,
    "max_new_tokens": module.MAX_NEW_TOKENS,
    "sampling_id": module.CACTUS_SAMPLING_ID,
    "stop_id": module.CACTUS_STOP_ID,
    "warmup_iterations": 1,
    "warmup_runs": 1,
    "iterations": 1,
    "runs": 1,
    "wall_ns_per_request": 1.0,
    "prefill_tokens_per_second": 1.0,
    "decode_tokens_per_second": 1.0,
    "phase_rate_semantics": module.PHASE_NONCOMPARABLE_REASON,
    "needle_package_version": module.NEEDLE_PACKAGE_VERSION,
    "needle_package_tree_sha256": module.NEEDLE_PACKAGE_TREE_SHA256,
    "needle_native_library_sha256": module.NEEDLE_NATIVE_LIBRARY_SHA256,
    "normalized_envelopes": [{"success": True}] * module.PROMPT_ROWS,
}
module.validate_reference(reference)
class FakeEngine:
    def reset(self):
        pass
    def complete(self, query, max_new_tokens):
        return {"success": True, "function_calls": [], "prefill_tps": 1.0,
                "decode_tps": 1.0, "peak_ram_mb": 1.0}

saved = {name: getattr(module, name) for name in (
    "validate_canonical_path", "validate_canonical_input", "load_requests",
    "validate_needle_package", "import_needle", "validate_needle_native_library")}
module.validate_canonical_path = lambda *args, **kwargs: None
module.validate_canonical_input = lambda *args, **kwargs: None
module.load_requests = lambda path: [("system", [], f"query-{index}")
                                    for index in range(module.PROMPT_ROWS)]
module.validate_needle_package = lambda *args, **kwargs: root / "needle"
module.import_needle = lambda *args, **kwargs: types.SimpleNamespace(
    Needle=lambda **kwargs: FakeEngine())
module.validate_needle_native_library = lambda *args, **kwargs: root / "libneedle.so"
live_record = module.run_reference(types.SimpleNamespace(
    model=str(root / "model.cact"), fixture=str(root / "fixture.tsv"),
    needle_root=str(root), staged=False, warmup_iterations=0, warmup_runs=0,
    iterations=1, runs=1))
assert live_record["needle_package_tree_sha256"] == module.NEEDLE_PACKAGE_TREE_SHA256
module.validate_reference(live_record)
for name, value in saved.items():
    setattr(module, name, value)
for key, value in (("runs", True), ("wall_ns_per_request", math.nan),
                   ("decode_tokens_per_second", 0.0),
                   ("sampling_id", module.EMEL_SAMPLING_ID),
                   ("needle_package_version", "2.0.7"),
                   ("needle_package_tree_sha256", "0" * 64),
                   ("needle_native_library_sha256", "f" * 64)):
    invalid = dict(reference)
    invalid[key] = value
    rejected(lambda invalid=invalid: module.validate_reference(invalid))

def emel_text(metric="1.0", runs="1", envelope=None):
    envelope = envelope or {"success": True}
    envelope_hex = module.canonical_envelope(envelope).encode().hex()
    rows = [
        f"# needle_request_envelope: workload_id={module.WORKLOAD_ID} row={row} hex={envelope_hex}"
        for row in range(module.PROMPT_ROWS)
    ]
    common = ("model_id=route_w4_qat_cact "
              "workload_id=needle_heldout_first4_greedy80_eos_v1 "
              "backend_id=emel_needle_request_serial route=serial "
              "fixture_id=tests/fixtures/cact/needle-heldout-prompts.tsv "
              "thread_count=1 thread_contract=single_thread "
              "prompt_rows=4 max_new_tokens=80 sampling_id=" + module.EMEL_SAMPLING_ID + " "
              "stop_id=" + module.EMEL_STOP_ID + " phase_tokens_per_batch=1 "
              "warmup_iterations=1 warmup_runs=1 phase_rate_semantics=" +
              module.PHASE_NONCOMPARABLE_REASON)
    for phase in ("wall", "prefill", "decode"):
        rows.append(f"# needle_graph: lane=emel case=x {common} phase={phase}")
        rows.append(f"x ns_per_op={metric} tokens_per_second={metric} iter=1 runs={runs}")
    return "\n".join(rows) + "\n"

bad_emel = root / "bad-emel.txt"
for text in (emel_text("nan"), emel_text("0"), emel_text("1", "33")):
    bad_emel.write_text(text)
    rejected(lambda: module.parse_emel(bad_emel))
for key, expected, wrong in (
        ("thread_contract", module.EMEL_THREAD_CONTRACT, "wrong_contract"),
        ("prompt_rows", str(module.PROMPT_ROWS), str(module.PROMPT_ROWS - 1)),
        ("max_new_tokens", str(module.MAX_NEW_TOKENS),
         str(module.MAX_NEW_TOKENS - 1))):
    bad_emel.write_text(
        emel_text().replace(f"{key}={expected}", f"{key}={wrong}"))
    rejected(lambda: module.parse_emel(bad_emel))

wrong_marker_masked_by_metric = root / "wrong-marker-masked-by-metric.txt"
wrong_marker_masked_by_metric.write_text(
    emel_text().replace("thread_contract=single_thread ",
                        "thread_contract=wrong_contract ").replace(
        " iter=1 runs=1", " thread_contract=single_thread iter=1 runs=1"))
rejected(lambda: module.parse_emel(wrong_marker_masked_by_metric))

marker_metric_collision = root / "marker-metric-collision.txt"
marker_metric_collision.write_text(
    emel_text().replace(" iter=1 runs=1",
                        " backend_id=emel_needle_request_serial iter=1 runs=1"))
rejected(lambda: module.parse_emel(marker_metric_collision))

unexpected_metric_metadata = root / "unexpected-metric-metadata.txt"
unexpected_metric_metadata.write_text(
    emel_text().replace(" iter=1 runs=1", " metadata=unexpected iter=1 runs=1"))
rejected(lambda: module.parse_emel(unexpected_metric_metadata))

duplicate_metric_key = root / "duplicate-metric-key.txt"
duplicate_metric_key.write_text(
    emel_text().replace(" iter=1 runs=1", " iter=1 iter=1 runs=1"))
rejected(lambda: module.parse_emel(duplicate_metric_key))

old_contract_emel = root / "old-contract-emel.txt"
old_contract_emel.write_text(
    emel_text().replace(
        "backend_id=emel_needle_request_serial route=serial ",
        "backend_id=emel_needle_request_parallel4 route=parallel4 ").replace(
        "thread_count=1 thread_contract=single_thread ",
        "thread_count=4 thread_contract=bounded_fork_join_3_workers_plus_owner "))
rejected(lambda: module.parse_emel(old_contract_emel))

good_emel = root / "good-emel.txt"
good_emel.write_text(emel_text())
parsed = module.parse_emel(good_emel)
assert parsed["backend_id"] == "emel_needle_request_serial"
assert parsed["thread_count"] == module.THREAD_COUNT == 1
large_envelope = {"success": True, "reasoning": "x" * 2048}
large_emel = root / "large-emel.txt"
large_emel.write_text(emel_text(envelope=large_envelope))
large_parsed = module.parse_emel(large_emel)
assert large_parsed["normalized_envelopes"] == [large_envelope] * module.PROMPT_ROWS
unordered_json = '{"success":true,"function_calls":[{"name":"route","arguments":{"z":1,"a":2}}]}'
unordered_emel = root / "unordered-emel.txt"
unordered_hex = unordered_json.encode().hex()
unordered_emel.write_text(emel_text().replace(
    module.canonical_envelope({"success": True}).encode().hex(), unordered_hex))
unordered_parsed = module.parse_emel(unordered_emel)
assert unordered_parsed["normalized_envelopes"][0]["function_calls"][0]["name"] == "route"

mismatch_reference = dict(reference)
mismatch_reference["normalized_envelopes"] = [{"success": False}] * module.PROMPT_ROWS
mismatch_reference_path = root / "mismatch-reference.json"
mismatch_reference_path.write_text(json.dumps(mismatch_reference))
with contextlib.redirect_stdout(io.StringIO()) as mismatch_output:
    module.compare(types.SimpleNamespace(
        emel_input=str(good_emel), reference_input=str(mismatch_reference_path)))
assert "output_parity=mismatch" in mismatch_output.getvalue()
assert "comparable=true" not in mismatch_output.getvalue()
assert "ratio=" not in mismatch_output.getvalue()
assert parsed["thread_contract"] == module.THREAD_CONTRACT == "single_thread"
reference_path = root / "reference.json"
reference_path.write_text(json.dumps(reference))
with contextlib.redirect_stdout(io.StringIO()) as exact_output:
    module.compare(types.SimpleNamespace(
        emel_input=str(good_emel), reference_input=str(reference_path)))
assert "output_parity=exact" in exact_output.getvalue()
assert "wall_comparison=noncomparable_closed_reference_contract" in exact_output.getvalue()
assert "comparable=true" not in exact_output.getvalue()
assert "ratio=" not in exact_output.getvalue()
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-cactus-boundary";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
#endif
}

TEST_CASE("needle cactus rejects substituted package before executing it") {
#if !defined(_WIN32)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-cactus-timeout";
  const std::filesystem::path needle_root = tmp_dir / "needle_root";
  const std::filesystem::path needle_package = needle_root / "needle";
  std::filesystem::create_directories(needle_package);
  const std::filesystem::path import_sentinel = tmp_dir / "imported.txt";
  write_file(needle_package / "__init__.py",
             "from pathlib import Path\nPath(r\"" +
                 import_sentinel.generic_string() +
                 "\").write_text(\"executed\")\n");
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path output_path = tmp_dir / "reference.json";
  const std::string command =
      "python3 -B " + quote_arg_posix(cactus_reference_driver_path().string()) +
      " run-reference --model " +
      quote_arg_posix((repo_root() / "tests" / "models" /
                       "route-w4-qat.cact").string()) +
      " --fixture " +
      quote_arg_posix((repo_root() / "tests" / "fixtures" / "cact" /
                       "needle-heldout-prompts.tsv").string()) +
      " --needle-root " + quote_arg_posix(needle_root.string()) +
      " --warmup-iterations 0 --warmup-runs 0 --iterations 1 --runs 1" +
      " --timeout-seconds 1 --output " + quote_arg_posix(output_path.string()) +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("Needle package tree SHA-256 mismatch") !=
        std::string::npos);
  CHECK_FALSE(std::filesystem::exists(import_sentinel));
  CHECK_FALSE(std::filesystem::exists(output_path));
#endif
}

TEST_CASE("needle cactus direct worker reentry is unavailable") {
#if !defined(_WIN32)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-worker-reentry";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path sentinel = tmp_dir / "imported.txt";
  const std::filesystem::path needle_root = tmp_dir / "needle-root";
  const std::filesystem::path needle_package = needle_root / "needle";
  std::filesystem::create_directories(needle_package);
  write_file(needle_package / "__init__.py",
             "from pathlib import Path\nPath(r\"" +
                 sentinel.generic_string() +
                 "\").write_text(\"executed\")\n");
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B " +
      quote_arg_posix(cactus_reference_driver_path().string()) +
      " run-reference-worker --staged --model ignored --fixture ignored" +
      " --needle-root " + quote_arg_posix(needle_root.string()) +
      " --output " + quote_arg_posix((tmp_dir / "output.json").string()) +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("invalid choice: 'run-reference-worker'") !=
        std::string::npos);
  CHECK_FALSE(std::filesystem::exists(sentinel));
#endif
}

TEST_CASE("needle cactus supervisor forks only over staged authenticated bytes") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import argparse
import hashlib
import importlib.util
import os
import pathlib
import sys
import tempfile

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

root = pathlib.Path(tempfile.mkdtemp())
needle_root = root / "needle-root"
package_root = needle_root / "needle"
package_root.mkdir(parents=True)
library = root / "libneedle.so"
model = root / "model.cact"
fixture = root / "fixture.tsv"
output = root / "reference.json"
library_bytes = b"authenticated native library"
model_bytes = b"authenticated model"
fixture_bytes = b"authenticated fixture"
library.write_bytes(library_bytes)
model.write_bytes(model_bytes)
fixture.write_bytes(fixture_bytes)
init_bytes = (
    "__version__ = '2.0.8'\n"
    "class Needle: pass\n"
    "def _library_path():\n"
    f"    return {str(library)!r}\n"
).encode()
(package_root / "__init__.py").write_bytes(init_bytes)
sha256 = lambda data: hashlib.sha256(data).hexdigest()
module.MODEL_SHA256 = sha256(model_bytes)
module.FIXTURE_SHA256 = sha256(fixture_bytes)
module.NEEDLE_PACKAGE_INIT_SHA256 = sha256(init_bytes)
module.NEEDLE_PACKAGE_TREE_SHA256 = module.sha256_python_tree(package_root)
module.NEEDLE_NATIVE_LIBRARY_SHA256 = sha256(library_bytes)
module.validate_canonical_path = lambda path, expected, name: None
os.environ["NEEDLE_TELEMETRY"] = "1"
os.environ["NEEDLE_TELEMETRY_URL"] = "https://attacker.invalid/collect"
os.environ.pop("DO_NOT_TRACK", None)

captured = {}
def fake_fork(args, staged_model, staged_fixture, staged_root,
              staged_library, timeout, environment):
    (package_root / "__init__.py").write_bytes(
        b"raise RuntimeError('swapped package executed')\n")
    model.write_bytes(b"swapped model")
    fixture.write_bytes(b"swapped fixture")
    library.write_bytes(b"swapped native library")
    staged_package = staged_root / "needle"
    assert staged_model != model
    assert staged_fixture != fixture
    assert staged_root != needle_root
    assert staged_model.read_bytes() == model_bytes
    assert staged_fixture.read_bytes() == fixture_bytes
    assert (staged_package / "__init__.py").read_bytes() == init_bytes
    assert staged_library.read_bytes() == library_bytes
    assert module.sha256_python_tree(
        staged_package, allow_native_library=staged_library
    ) == module.NEEDLE_PACKAGE_TREE_SHA256
    assert module.sha256_file(staged_library) == module.NEEDLE_NATIVE_LIBRARY_SHA256
    assert environment["NEEDLE_LIB_PATH"] == str(staged_library)
    assert environment["NEEDLE_TELEMETRY"] == "0"
    assert environment["DO_NOT_TRACK"] == "1"
    assert "NEEDLE_TELEMETRY_URL" not in environment
    assert not (set(module.INJECTION_ENVIRONMENT_VARIABLES) &
                (set(environment) - {"NEEDLE_LIB_PATH"}))
    pathlib.Path(args.output).write_text('{"ok": true}\n')
    captured["stage"] = staged_root.parent

module.run_forked_reference = fake_fork
module.run_reference_subprocess(argparse.Namespace(
    timeout_seconds=5, model=str(model), fixture=str(fixture),
    needle_root=str(needle_root), warmup_iterations=0, warmup_runs=0,
    iterations=1, runs=1, output=str(output)))
assert not captured["stage"].exists()
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-staged-authentication";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}

TEST_CASE("needle cactus authenticated child forces telemetry off") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = pathlib.Path(tempfile.mkdtemp())
output = root / "output.json"
args = argparse.Namespace(warmup_iterations=0, warmup_runs=0, iterations=1,
                          runs=1, output=str(output))
paths = [root / name for name in ("model", "fixture", "package", "library")]
for path in paths:
    path.mkdir() if path.name == "package" else path.write_bytes(b"x")

def capture_environment(_args):
    return {
        "needle_telemetry": os.environ.get("NEEDLE_TELEMETRY"),
        "do_not_track": os.environ.get("DO_NOT_TRACK"),
        "telemetry_url_present": "NEEDLE_TELEMETRY_URL" in os.environ,
    }

module.run_reference = capture_environment
module.run_forked_reference(
    args, paths[0], paths[1], paths[2], paths[3], 10,
    {"NEEDLE_TELEMETRY": "1", "DO_NOT_TRACK": "",
     "NEEDLE_TELEMETRY_URL": "https://attacker.invalid/collect"})

record = json.loads(output.read_text())
assert record == {
    "needle_telemetry": "0",
    "do_not_track": "1",
    "telemetry_url_present": False,
}
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-telemetry-isolation";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}
TEST_CASE("needle cactus output ignores predictable sibling symlink") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import importlib.util
import pathlib
import sys
import tempfile

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = pathlib.Path(tempfile.mkdtemp())
output = root / "reference.json"
sentinel = root / "sentinel"
sentinel.write_text("unchanged")
(root / "reference.json.tmp").symlink_to(sentinel)
module.write_reference_output({"ok": True}, output)
assert output.read_text() == '{"ok": true}\n'
assert sentinel.read_text() == "unchanged"
assert (root / "reference.json.tmp").is_symlink()
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-output-symlink";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}


TEST_CASE("needle cactus supervisor reaps live child on timeout exception and signal") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import argparse
import importlib.util
import os
import pathlib
import signal
import sys
import tempfile
import time

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = pathlib.Path(tempfile.mkdtemp())
args = argparse.Namespace(warmup_iterations=0, warmup_runs=0, iterations=1,
                          runs=1, output=str(root / "output.json"))
staged = root / "staged"
staged.mkdir()
model = staged / "model"
fixture = staged / "fixture"
package = staged / "package"
library = staged / "library"
for path in (model, fixture, library):
    path.write_bytes(b"x")
package.mkdir()

def assert_reaped(pid, failure):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(failure)

original_handlers = {sig: signal.getsignal(sig)
                     for sig in module.SUPERVISOR_SIGNALS}
real_monotonic = time.monotonic
real_sleep = time.sleep

timeout_worker_path = root / "timeout-worker.pid"
def timeout_worker(_args):
    timeout_worker_path.write_text(str(os.getpid()))
    real_sleep(60)
module.run_reference = timeout_worker
timeout_now = 0
def timeout_monotonic():
    global timeout_now
    timeout_now += 1
    if timeout_now == 2:
        deadline = real_monotonic() + 5
        while not timeout_worker_path.exists() and real_monotonic() < deadline:
            real_sleep(0.001)
    return timeout_now
module.time.monotonic = timeout_monotonic
module.time.sleep = lambda _seconds: None
try:
    module.run_forked_reference(args, model, fixture, package, library, 1, {})
    raise AssertionError("timeout was swallowed")
except SystemExit as exc:
    assert "exceeded 1s timeout" in str(exc)
finally:
    module.time.monotonic = real_monotonic
    module.time.sleep = real_sleep
assert_reaped(int(timeout_worker_path.read_text()), "timeout left worker alive")
assert {sig: signal.getsignal(sig) for sig in module.SUPERVISOR_SIGNALS} == original_handlers

exception_worker_path = root / "exception-worker.pid"
def exception_worker(_args):
    exception_worker_path.write_text(str(os.getpid()))
    real_sleep(60)
module.run_reference = exception_worker
calls = 0
def parent_exception():
    global calls
    calls += 1
    if calls == 1:
        return real_monotonic()
    deadline = real_monotonic() + 5
    while not exception_worker_path.exists() and real_monotonic() < deadline:
        real_sleep(0.001)
    raise RuntimeError("parent wait failure")
module.time.monotonic = parent_exception
try:
    module.run_forked_reference(args, model, fixture, package, library, 10, {})
    raise AssertionError("parent exception was swallowed")
except RuntimeError as exc:
    assert str(exc) == "parent wait failure"
finally:
    module.time.monotonic = real_monotonic
assert_reaped(int(exception_worker_path.read_text()), "exception left worker alive")
assert {sig: signal.getsignal(sig) for sig in module.SUPERVISOR_SIGNALS} == original_handlers

signal_worker_path = root / "signal-worker.pid"
supervisor = os.fork()
if supervisor == 0:
    def signal_worker(_args):
        signal_worker_path.write_text(str(os.getpid()))
        real_sleep(60)
    module.run_reference = signal_worker
    module.run_forked_reference(args, model, fixture, package, library, 10, {})
    os._exit(0)
deadline = real_monotonic() + 5
while not signal_worker_path.exists() and real_monotonic() < deadline:
    real_sleep(0.001)
assert signal_worker_path.exists(), "signal worker did not start"
os.kill(supervisor, signal.SIGTERM)
waited, status = os.waitpid(supervisor, 0)
assert waited == supervisor
assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 128 + signal.SIGTERM
assert_reaped(int(signal_worker_path.read_text()), "signal left worker alive")
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-supervisor-lifecycle";
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}

TEST_CASE("needle cactus supervisor uses an explicit safe worker environment") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import argparse
import importlib.util
import os
import pathlib
import sys
import tempfile

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in module.WORKER_ENVIRONMENT_ALLOWLIST:
    os.environ.pop(name, None)
for name in module.INJECTION_ENVIRONMENT_VARIABLES:
    os.environ[name] = "injected"
os.environ.update(HOME="/safe-home", XDG_CACHE_HOME="/safe-cache",
                  TMPDIR="/safe-temp", LANG="C.UTF-8", LC_ALL="C",
                  NEEDLE_THREADS="1", UNRELATED_SECRET="must-not-pass")
root = pathlib.Path(tempfile.mkdtemp())
needle_root = root / "needle-root"
(needle_root / "needle").mkdir(parents=True)
model = root / "model"
fixture = root / "fixture"
model.write_bytes(b"model")
fixture.write_bytes(b"fixture")
module.copy_authenticated_file = lambda source, destination, name: destination.write_bytes(source.read_bytes())
module.stage_needle_package = lambda source, destination: destination / "needle"
module.import_needle = lambda root, package: object()
module.stage_needle_native_library = lambda needle, package: package / "libneedle.so"
module.validate_canonical_input = lambda path, expected, name: None
module.validate_canonical_path = lambda path, expected, name: None
module.validate_needle_package = lambda root, **kwargs: root / "needle"
captured = {}
def fake_fork(args, staged_model, staged_fixture, staged_root,
              staged_library, timeout, environment):
    captured.update(timeout=timeout, environment=environment,
                    staged_library=staged_library)
module.run_forked_reference = fake_fork
module.run_reference_subprocess(argparse.Namespace(
    timeout_seconds=5, model=str(model), fixture=str(fixture),
    needle_root=str(needle_root), warmup_iterations=0, warmup_runs=0,
    iterations=1, runs=1, output=str(root / "output.json")))
assert captured["timeout"] == 5
worker_env = captured["environment"]
assert worker_env == {
    "HOME": "/safe-home", "XDG_CACHE_HOME": "/safe-cache",
    "TMPDIR": "/safe-temp", "LANG": "C.UTF-8", "LC_ALL": "C",
    "NEEDLE_THREADS": "1", "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "NEEDLE_LIB_PATH": str(captured["staged_library"]),
}
assert not (set(module.INJECTION_ENVIRONMENT_VARIABLES) &
            (set(worker_env) - {"NEEDLE_LIB_PATH"}))
assert "UNRELATED_SECRET" not in worker_env
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-worker-environment";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}
TEST_CASE("needle cactus aggregates iterations within runs before median") {
#if !defined(_WIN32)
  const std::string program = R"PY(
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("cactus_reference", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.median_run_means([[1.0, 99.0], [60.0, 60.0], [70.0, 70.0]],
                               "sample") == 60.0
)PY";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-run-aggregation";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      "python3 -I -S -B -c " + quote_arg_posix(program) + " " +
      quote_arg_posix(cactus_reference_driver_path().string()) + " > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.empty());
#endif
}


TEST_CASE("needle canonical compare has pinned model and retokenizes fixture") {
  const std::string wrapper =
      read_file(repo_root() / "scripts" / "bench.sh");
  const std::string driver = read_file(cactus_reference_driver_path());
  const std::string graph = read_file(
      repo_root() / "tools" / "bench" / "model" / "needle" /
      "graph_bench.cpp");
  CHECK(wrapper.find("EMEL_BENCH_NEEDLE_MODEL is unsupported") !=
        std::string::npos);
  CHECK(driver.find("MODEL_SHA256 =") != std::string::npos);
  CHECK(driver.find("FIXTURE_SHA256 =") != std::string::npos);
  CHECK(driver.find("NEEDLE_PACKAGE_TREE_SHA256 =") != std::string::npos);
  CHECK(driver.find("NEEDLE_NATIVE_LIBRARY_SHA256 =") != std::string::npos);
  CHECK(driver.find("NEEDLE_PACKAGE_VERSION = \"2.0.8\"") != std::string::npos);
  CHECK(wrapper.find("NEEDLE_LIB_PATH is unsupported") != std::string::npos);
  CHECK(wrapper.find("NEEDLE_PYTHON_SHA256=") != std::string::npos);
  CHECK(wrapper.find("validate_needle_python \"${EMEL_BENCH_NEEDLE_PYTHON}\"") !=
        std::string::npos);
  CHECK(wrapper.find("NEEDLE_AUTHENTICATED_EXEC=") != std::string::npos);
  CHECK(wrapper.find("needle_authenticated_exec") != std::string::npos);
  CHECK(wrapper.find("NEEDLE_PYTHON_STAGE_DIR=") == std::string::npos);
  CHECK(wrapper.find("sha256sum \"$staged_python\"") == std::string::npos);
  CHECK(wrapper.find("local python_executable=\"$NEEDLE_PYTHON_EXECUTABLE\"") !=
        std::string::npos);
  CHECK(wrapper.find("validate_needle_python \"$python_executable\"") ==
        std::string::npos);
  CHECK(graph.find("request_fixture_token_id_mismatch") != std::string::npos);
  CHECK(graph.find("request.text = row.prompt") != std::string::npos);
  CHECK(graph.find("actual != row.token_ids") != std::string::npos);
  CHECK(driver.find("cactus_public_sampling_unverified") != std::string::npos);
  CHECK(driver.find("cactus_public_stop_unverified") != std::string::npos);
  CHECK(driver.find("closed_reference_request_contract_missing_formatter_constraint_sampling_stop") !=
        std::string::npos);
  CHECK(driver.find("comparable=true") == std::string::npos);
  CHECK(driver.find("wall_comparison=noncomparable_closed_reference_contract") !=
        std::string::npos);
  CHECK(driver.find("ratio=") == std::string::npos);
  CHECK(graph.find("phase_rate_semantics=closed_reference_phase_contract_missing_token_counts_and_timestamps") !=
        std::string::npos);
  CHECK(wrapper.find("EMEL_BENCH_NEEDLE_TIMEOUT_SECONDS") !=
        std::string::npos);
  CHECK(driver.find("worker_environment({\"NEEDLE_LIB_PATH\": str(staged_library)})") !=
        std::string::npos);
  CHECK(driver.find("median_run_means") != std::string::npos);
  CHECK(graph.find("proof_status=measurement_only") != std::string::npos);
  CHECK(graph.find("out.comparable = false;") != std::string::npos);
  CHECK(wrapper.find("resolve_needle_python") != std::string::npos);
  CHECK(wrapper.find("readlink -f \"$python_executable\" 2>/dev/null") !=
        std::string::npos);
}
TEST_CASE("needle compare default build still provisions authenticated exec") {
  const std::string wrapper = read_file(repo_root() / "scripts" / "bench.sh");
  const std::size_t configure_begin =
      wrapper.find("configure_bench_build() {");
  REQUIRE(configure_begin != std::string::npos);
  const std::size_t configure_end =
      wrapper.find("\nupdate_snapshot_baseline() {", configure_begin);
  REQUIRE(configure_end != std::string::npos);
  const std::string configure =
      wrapper.substr(configure_begin, configure_end - configure_begin);
  CHECK(configure.find("if [[ \"$SUITE_FILTER\" == \"needle_graph\" ]]") !=
        std::string::npos);
  CHECK(configure.find("if [[ \"$build_suite_filter\" == \"needle_graph\" ]]") ==
        std::string::npos);
  CHECK(configure.find("bench_runner needle_authenticated_exec") !=
        std::string::npos);
  CHECK(configure.find(
            "NEEDLE_AUTHENTICATED_EXEC=\"$build_dir/needle_authenticated_exec\"") !=
        std::string::npos);
}


TEST_CASE("needle authenticated exec resists source replacement after authentication") {
#if defined(__linux__)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-authenticated-exec-race";
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path configured = tmp_dir / "python";
  const std::filesystem::path replacement = tmp_dir / "replacement";
  const std::filesystem::path original_marker = tmp_dir / "original-ran";
  const std::filesystem::path replacement_marker = tmp_dir / "replacement-ran";
  const std::filesystem::path ready = tmp_dir / "ready";
  const std::filesystem::path proceed = tmp_dir / "continue";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::filesystem::path status_path = tmp_dir / "status.txt";
  std::filesystem::copy_file("/usr/bin/python3.12", configured);
  std::filesystem::copy_file("/bin/false", replacement);
  make_executable(configured);
  make_executable(replacement);
  const std::string digest_command =
      "sha256sum " + quote_arg_posix(configured.string()) +
      " | cut -d' ' -f1";
  std::array<char, 128> digest_buffer{};
  FILE *digest_pipe = popen(digest_command.c_str(), "r");
  REQUIRE(digest_pipe != nullptr);
  REQUIRE(std::fgets(digest_buffer.data(),
                     static_cast<int>(digest_buffer.size()), digest_pipe) != nullptr);
  REQUIRE(pclose(digest_pipe) == 0);
  std::string digest = digest_buffer.data();
  while (!digest.empty() && (digest.back() == '\n' || digest.back() == '\r')) {
    digest.pop_back();
  }

  const std::string helper =
      quote_arg_posix(needle_authenticated_exec_test_path().string());
  const std::string command =
      "set -eu; EMEL_AUTHENTICATED_EXEC_TEST_READY=" +
      quote_arg_posix(ready.string()) +
      " EMEL_AUTHENTICATED_EXEC_TEST_CONTINUE=" + quote_arg_posix(proceed.string()) +
      " " + helper + " " + quote_arg_posix(configured.string()) + " " +
      quote_arg_posix(digest) + " -- " + quote_arg_posix(configured.string()) +
      " -I -S -B -c " +
      quote_arg_posix("open('" + original_marker.string() +
                      "','w').write('original')") +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string()) + " & child=$!; "
      "while [ ! -e " + quote_arg_posix(ready.string()) + " ]; do sleep 0.01; done; "
      "mv -f " + quote_arg_posix(replacement.string()) + " " +
      quote_arg_posix(configured.string()) + "; : > " +
      quote_arg_posix(proceed.string()) + "; wait $child; printf '%s' $? > " +
      quote_arg_posix(status_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(read_file(status_path) == "0");
  CHECK(capture.stderr_text.empty());
  CHECK(std::filesystem::exists(original_marker));
  CHECK_FALSE(std::filesystem::exists(replacement_marker));
#endif
}

TEST_CASE("needle authenticated exec rejects the wrong interpreter hash") {
#if defined(__linux__)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-authenticated-exec-wrong-hash";
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path configured = tmp_dir / "python";
  const std::filesystem::path marker = tmp_dir / "ran";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  write_file(configured,
             "#!/bin/sh\nprintf ran > " + quote_arg_posix(marker.string()) + "\n");
  make_executable(configured);
  const process_capture capture = run_command_capture(
      quote_arg_posix(needle_authenticated_exec_test_path().string()) + " " +
          quote_arg_posix(configured.string()) + " " + std::string(64u, '0') +
          " -- " + quote_arg_posix(configured.string()) + " > " +
          quote_arg_posix(stdout_path.string()) + " 2> " +
          quote_arg_posix(stderr_path.string()),
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("configured Needle Python SHA-256 mismatch") !=
        std::string::npos);
  CHECK_FALSE(std::filesystem::exists(marker));
#endif
}

TEST_CASE("needle authenticated exec closes interpreter memfd across exec") {
#if defined(__linux__)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-authenticated-exec-cloexec";
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string program =
      "import os,pathlib; leaked=[]; "
      "[(leaked.append(str(p)) if 'emel-needle-python' in "
      "os.readlink(p) else None) for p in pathlib.Path('/proc/self/fd').iterdir() "
      "if p.exists()]; assert not leaked, leaked; print('cloexec-ok')";
  const process_capture capture = run_command_capture(
      quote_arg_posix(needle_authenticated_exec_test_path().string()) + " " +
          quote_arg_posix("/usr/bin/python3.12") + " " +
          "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118" +
          " -- /usr/bin/python3.12 -I -S -B -c " + quote_arg_posix(program) +
          " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
          quote_arg_posix(stderr_path.string()),
      stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text == "cloexec-ok\n");
  CHECK(capture.stderr_text.empty());
#endif
}

TEST_CASE("needle authenticated exec accepts every allowlisted environment variable") {
#if defined(__linux__)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-authenticated-exec-full-environment";
  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string environment =
      "HOME=/tmp XDG_CACHE_HOME=/tmp TMPDIR=/tmp TMP=/tmp TEMP=/tmp "
      "LANG=C LANGUAGE=C LC_ALL=C LC_CTYPE=C LC_NUMERIC=C LC_TIME=C "
      "LC_COLLATE=C LC_MONETARY=C LC_MESSAGES=C LC_PAPER=C LC_NAME=C "
      "LC_ADDRESS=C LC_TELEPHONE=C LC_MEASUREMENT=C LC_IDENTIFICATION=C "
      "NEEDLE_THREADS=1 ";
  const std::string program =
      "import os; names=('HOME','XDG_CACHE_HOME','TMPDIR','TMP','TEMP','LANG',"
      "'LANGUAGE','LC_ALL','LC_CTYPE','LC_NUMERIC','LC_TIME','LC_COLLATE',"
      "'LC_MONETARY','LC_MESSAGES','LC_PAPER','LC_NAME','LC_ADDRESS',"
      "'LC_TELEPHONE','LC_MEASUREMENT','LC_IDENTIFICATION','NEEDLE_THREADS'); "
      "assert all(name in os.environ for name in names); print('full-env-ok')";
  const process_capture capture = run_command_capture(
      environment + quote_arg_posix(needle_authenticated_exec_test_path().string()) +
          " /usr/bin/python3.12 "
          "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118" +
          " -- /usr/bin/python3.12 -I -S -B -c " + quote_arg_posix(program) +
          " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
          quote_arg_posix(stderr_path.string()),
      stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text == "full-env-ok\n");
  CHECK(capture.stderr_text.empty());
#endif
}

TEST_CASE("needle compare wrapper rejects model, library override, and unbounded counts") {
#if !defined(_WIN32)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-wrapper-boundaries";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path fake_python = tmp_dir / "python";
  const std::filesystem::path fake_needle = tmp_dir / "needle";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  write_file(fake_python, "#!/bin/sh\nexit 0\n");
  make_executable(fake_python);
  std::filesystem::create_directories(fake_needle);
  const std::string base =
      needle_clean_environment_prefix() +
      "EMEL_BENCH_NEEDLE_PYTHON=" + quote_arg_posix(fake_python.string()) +
      " EMEL_BENCH_NEEDLE_ROOT=" + quote_arg_posix(fake_needle.string()) + " ";
  const std::string wrapper =
      quote_arg_posix((repo_root() / "scripts" / "bench.sh").string()) +
      " --compare --suite=needle_graph --system > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  process_capture capture = run_command_capture(
      base + "EMEL_BENCH_NEEDLE_REQUEST_RUNS=26 " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find(
            "EMEL_BENCH_NEEDLE_REQUEST_RUNS must be an integer in [1, 25]") !=
        std::string::npos);
  capture = run_command_capture(
      base + "EMEL_BENCH_NEEDLE_REQUEST_WARMUP_RUNS=0 " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find(
            "EMEL_BENCH_NEEDLE_REQUEST_WARMUP_RUNS must be an integer in [1, 25]") !=
        std::string::npos);
  capture = run_command_capture(
      base + "EMEL_BENCH_NEEDLE_REQUEST_ITERS=33 " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find(
            "EMEL_BENCH_NEEDLE_REQUEST_ITERS must be an integer in [1, 32]") !=
        std::string::npos);
  capture = run_command_capture(
      base + "EMEL_BENCH_NEEDLE_MODEL=/tmp/substitute " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("EMEL_BENCH_NEEDLE_MODEL is unsupported") !=
        std::string::npos);
  capture = run_command_capture(
      base + "NEEDLE_LIB_PATH=/tmp/substitute " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("NEEDLE_LIB_PATH is unsupported") !=
        std::string::npos);
  capture = run_command_capture(base + wrapper, stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("Needle Python SHA-256 mismatch") !=
        std::string::npos);
  capture = run_command_capture(
      needle_clean_environment_prefix() +
          "EMEL_BENCH_NEEDLE_PYTHON=" + quote_arg_posix(fake_needle.string()) +
          " EMEL_BENCH_NEEDLE_ROOT=" + quote_arg_posix(fake_needle.string()) +
          " " + wrapper,
      stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find(
            "cannot resolve canonical regular-file Needle Python executable") !=
        std::string::npos);
#endif
}

TEST_CASE("needle compare wrapper count contract matches runner semantics") {
  const std::string wrapper = read_file(repo_root() / "scripts" / "bench.sh");
  CHECK(wrapper.find("NEEDLE_REQUEST_MAX_ITERATIONS=32") !=
        std::string::npos);
  CHECK(wrapper.find("NEEDLE_REQUEST_MAX_RUNS=25") != std::string::npos);
  CHECK(wrapper.find(
            "\"$DEFAULT_NEEDLE_REQUEST_RUNS\" 1 \"$NEEDLE_REQUEST_MAX_RUNS\"") !=
        std::string::npos);
  CHECK(wrapper.find(
            "\"$DEFAULT_NEEDLE_REQUEST_WARMUP_RUNS\" 1 \"$NEEDLE_REQUEST_MAX_RUNS\"") !=
        std::string::npos);
  CHECK(wrapper.find(
            "\"$DEFAULT_NEEDLE_REQUEST_ITERS\" 1 \"$NEEDLE_REQUEST_MAX_ITERATIONS\"") !=
        std::string::npos);
  CHECK(wrapper.find(
            "\"$DEFAULT_NEEDLE_REQUEST_WARMUP_ITERS\" 0 \"$NEEDLE_REQUEST_MAX_ITERATIONS\"") !=
        std::string::npos);
}
TEST_CASE("needle compare wrapper rejects injection before launching Python") {
#if !defined(_WIN32)
  const std::array<std::string_view, 12> injection_variables = {
      "LD_PRELOAD",
      "LD_LIBRARY_PATH",
      "LD_AUDIT",
      "DYLD_LIBRARY_PATH",
      "DYLD_INSERT_LIBRARIES",
      "DYLD_FRAMEWORK_PATH",
      "DYLD_FALLBACK_LIBRARY_PATH",
      "DYLD_FALLBACK_FRAMEWORK_PATH",
      "PYTHONPATH",
      "PYTHONHOME",
      "PYTHONSTARTUP",
      "PYTHONINSPECT",
  };
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-wrapper-injection";
  const std::filesystem::path fake_needle = tmp_dir / "needle";
  const std::filesystem::path sentinel = tmp_dir / "python-launched.txt";
  const std::filesystem::path fake_python = tmp_dir / "python";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::filesystem::create_directories(fake_needle);
  write_file(fake_python,
             "#!/bin/sh\nprintf launched > " +
                 quote_arg_posix(sentinel.string()) + "\nexit 0\n");
  make_executable(fake_python);

  const std::string base =
      "EMEL_BENCH_NEEDLE_PYTHON=" + quote_arg_posix(fake_python.string()) +
      " EMEL_BENCH_NEEDLE_ROOT=" + quote_arg_posix(fake_needle.string()) + " ";
  const std::string wrapper =
      quote_arg_posix((repo_root() / "scripts" / "bench.sh").string()) +
      " --compare --suite=needle_graph --system > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());

  for (const std::string_view variable : injection_variables) {
    std::filesystem::remove(sentinel);
    const process_capture capture = run_command_capture(
        needle_clean_environment_prefix() + std::string{variable} +
            "=injected " + base + wrapper,
        stdout_path, stderr_path);
    CHECK(capture.exit_code != 0);
    CHECK(capture.stderr_text.find(
              "error: dynamic-loader/Python injection variable is set: " +
              std::string{variable}) != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(sentinel));
  }
#endif
}


TEST_CASE("needle graph emits live request metadata without recorded rows") {
#if defined(__x86_64__) || defined(_M_X64)
  const process_capture snapshot =
      run_needle_graph_bench_capture("emel", "needle-graph-emel");
  REQUIRE(snapshot.exit_code == 0);
  CHECK(snapshot.stderr_text.find("error:") == std::string::npos);
  CHECK(snapshot.stdout_text.find("libneedle-recorded") == std::string::npos);
  CHECK(snapshot.stdout_text.find("measurement_only") == std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "workload_id=needle_heldout_first4_greedy80_eos_v1") !=
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "reference=live_cactus_native phase=wall ") != std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "fixture_id=tests/fixtures/cact/needle-heldout-prompts.tsv") !=
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "backend_id=emel_needle_request_serial route=serial") !=
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "thread_count=1"
            " thread_contract=single_thread"
            " prompt_rows=4") != std::string::npos);
  CHECK(snapshot.stdout_text.find("emel_needle_request_parallel4") ==
        std::string::npos);
  const std::string graph_source = read_file(
      repo_root() / "tools" / "bench" / "model" / "needle" /
      "graph_bench.cpp");
  CHECK(graph_source.find("needle::request::sm adapter") != std::string::npos);
  CHECK(graph_source.find("run_request_batch(adapter, rows)") !=
        std::string::npos);
  CHECK(graph_source.find("run_request_batch<needle::graph::sm>") ==
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "max_new_tokens=80 sampling_id=greedy_argmax_v1 "
            "stop_id=eos_or_max80_v1") != std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "needle/graph/request_heldout_first4_greedy80/prefill ") !=
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "needle/graph/request_heldout_first4_greedy80/decode ") !=
        std::string::npos);
  CHECK(snapshot.stdout_text.find(
            "phase_rate_semantics=closed_reference_phase_contract_missing_token_counts_and_timestamps") !=
        std::string::npos);
  const std::string prefill_line = find_line_with_prefix(
      snapshot.stdout_text,
      "needle/graph/request_heldout_first4_greedy80/prefill ns_per_op=");
  REQUIRE_FALSE(prefill_line.empty());
  const double prefill_ns = parse_named_double(prefill_line, "ns_per_op");
  const double prefill_tps =
      parse_named_double(prefill_line, "tokens_per_second");
  REQUIRE(prefill_ns > 0.0);
  CHECK(prefill_tps * prefill_ns / 1000000000.0 ==
        doctest::Approx(336.75).epsilon(0.0001));
  CHECK(snapshot.stdout_text.find("phase_tokens_per_batch=1347 ") !=
        std::string::npos);
#endif
}

TEST_CASE("needle graph compare wrapper hard fails without live reference") {
#if !defined(_WIN32)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-missing-reference";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string command =
      needle_clean_environment_prefix() +
      "env -u EMEL_BENCH_NEEDLE_PYTHON -u EMEL_BENCH_NEEDLE_ROOT " +
      quote_arg_posix((repo_root() / "scripts" / "bench.sh").string()) +
      " --compare --suite=needle_graph --system > " +
      quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);
  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find("EMEL_BENCH_NEEDLE_PYTHON is required") !=
        std::string::npos);
#endif
}

TEST_CASE(
    "bench_runner generation compare keeps bounded maintained Liquid fixture") {
  const process_capture capture = run_generation_bench_compare_capture();
  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("# generation_architecture: lfm2") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("# generation_formatter_contract:") !=
        std::string::npos);
  CHECK(capture.stdout_text.find(
            "# generation_threading: applies_to=generated_generation_row "
            "case=" +
            std::string{k_bounded_generation_case_name} +
            "/single benchmark_lane=single emel_thread_count=1 "
            "reference_thread_count=1") != std::string::npos);
  CHECK(capture.stdout_text.find(
            "# generation_threading: applies_to=generated_generation_row "
            "case=" +
            std::string{k_bounded_generation_case_name} +
            "/multithreaded benchmark_lane=multithreaded emel_thread_count=8 "
            "reference_thread_count=") != std::string::npos);
  CHECK(capture.stdout_text.find("# generation_stage_probe: case=" +
                                 std::string{k_bounded_generation_case_name}) !=
        std::string::npos);
  CHECK(capture.stdout_text.find("emel_prefill_linear_probe_ns=") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("reference_prefill_attention_probe_ns=") !=
        std::string::npos);

  CHECK(capture.stdout_text.find(k_bounded_generation_case_name) !=
        std::string::npos);
  CHECK(capture.stdout_text.find("tokens/s") != std::string::npos);
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
  const std::string main_source =
      read_file(repo_root() / "tools" / "bench" / "bench_main.cpp");
  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");

  CHECK(main_source.find("run_bench_cli(argc, argv)") != std::string::npos);
  CHECK(main_source.find("default_test_cases") == std::string::npos);
  CHECK(main_source.find("run_benchmarks") == std::string::npos);
  CHECK(main_source.find("print_compare") == std::string::npos);
  CHECK(runner_source.find(
            "int emel::bench::run_bench_cli(int argc, char **argv)") !=
        std::string::npos);
  CHECK(runner_source.find("EMEL_BENCH_ITERS") != std::string::npos);
  CHECK(runner_source.find("print_compare") != std::string::npos);
}

TEST_CASE("bench runner contract serializes requests and results for a process "
          "seam") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::compare;
  request.suite = "generation";
  request.cfg.iterations = 17u;
  request.cfg.runs = 3u;
  request.cfg.warmup_iterations = 5u;
  request.cfg.warmup_runs = 1u;
  request.generation_jsonl = true;

  const std::string serialized = emel::bench::serialize_runner_request(request);
  CHECK(serialized.find("schema=bench_runner_request/v1\n") !=
        std::string::npos);
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
  const std::string negative_result_text =
      emel::bench::serialize_runner_result(result);
  CHECK(negative_result_text.find("exit_code=-1\n") != std::string::npos);

  emel::bench::runner_result parsed_negative_result = {};
  CHECK(emel::bench::parse_runner_result(negative_result_text,
                                         parsed_negative_result));
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
  const auto measured =
      emel::bench::measure_case("bench/zero_cfg", cfg, [&]() { ++calls; });

  CHECK(calls == 1u);
  CHECK(measured.iterations == 1u);
  CHECK(measured.runs == 1u);
}

TEST_CASE("benchmark run setup executes outside timing for every run") {
  emel::bench::config cfg = {};
  cfg.iterations = 3u;
  cfg.runs = 5u;
  cfg.warmup_iterations = 2u;
  cfg.warmup_runs = 1u;
  std::uint32_t setup_calls = 0u;
  std::uint32_t measured_calls = 0u;
  std::uint32_t calls_since_setup = 0u;
  bool fixed_run_extent = true;

  const auto measured = emel::bench::measure_case_with_run_setup(
      "bench/run_setup", cfg,
      [&]() {
        if (setup_calls > 0u) {
          const std::uint32_t expected_calls =
              setup_calls == 1u ? cfg.warmup_iterations : cfg.iterations;
          fixed_run_extent =
              fixed_run_extent && calls_since_setup == expected_calls;
        }
        ++setup_calls;
        calls_since_setup = 0u;
      },
      [&]() {
        ++measured_calls;
        ++calls_since_setup;
      });

  CHECK(setup_calls == cfg.warmup_runs + cfg.runs);
  CHECK(measured_calls ==
        cfg.warmup_iterations * cfg.warmup_runs + cfg.iterations * cfg.runs);
  CHECK(calls_since_setup == cfg.iterations);
  CHECK(fixed_run_extent);
  CHECK(measured.iterations == cfg.iterations);
  CHECK(measured.runs == cfg.runs);
}

TEST_CASE("bench runner contract rejects malformed process payloads") {
  emel::bench::runner_request request = {};
  CHECK_FALSE(emel::bench::parse_runner_request(
      "schema=bench_runner_request/v1\n", request));
  CHECK_FALSE(
      emel::bench::parse_runner_request("schema=bench_runner_request/v1\n"
                                        "mode=unknown\n"
                                        "suite=generation\n"
                                        "iterations=1\n"
                                        "runs=1\n"
                                        "warmup_iterations=0\n"
                                        "warmup_runs=0\n"
                                        "generation_jsonl=0\n"
                                        "diarization_jsonl=0\n",
                                        request));
  CHECK_FALSE(
      emel::bench::parse_runner_request("schema=bench_runner_request/v1\n"
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
  CHECK_FALSE(emel::bench::parse_runner_result(
      "schema=bench_runner_result/v1\n", result));
  CHECK_FALSE(emel::bench::parse_runner_result(
      "schema=bench_runner_result/v1\nexit_code=bad\n", result));
}

TEST_CASE("bench runner process seam executes a serialized request through the "
          "live binary") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::emel;
  request.suite = "generation";
  request.cfg.iterations = 1u;
  request.cfg.runs = 1u;
  request.cfg.warmup_iterations = 0u;
  request.cfg.warmup_runs = 0u;

  std::string result_text;
  const process_capture capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(request), "process-seam-generation",
      result_text);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("# benchmark_config:") != std::string::npos);
  CHECK(capture.stdout_text.find("generation/preloaded_request/") !=
        std::string::npos);

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 0);
  CHECK(result.error_kind.empty());
  CHECK(result.error_message.empty());
}

TEST_CASE("bench runner process seam writes deterministic errors for malformed "
          "payloads") {
  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture("schema=bench_runner_request/v1\n",
                                     "process-seam-malformed", result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("parse") != std::string::npos);
}

TEST_CASE(
    "bench runner process seam writes deterministic errors for unknown modes") {
  std::string result_text;
  const process_capture capture =
      run_serialized_request_capture("schema=bench_runner_request/v1\n"
                                     "mode=unknown\n"
                                     "suite=generation\n"
                                     "iterations=1\n"
                                     "runs=1\n"
                                     "warmup_iterations=0\n"
                                     "warmup_runs=0\n"
                                     "generation_jsonl=0\n"
                                     "diarization_jsonl=0\n",
                                     "process-seam-unknown-mode", result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("parse") != std::string::npos);
}

TEST_CASE("bench runner process seam writes deterministic errors for unknown "
          "suites") {
  emel::bench::runner_request request = {};
  request.mode = emel::bench::runner_mode::emel;
  request.suite = "missing_suite";
  request.cfg.iterations = 1u;
  request.cfg.runs = 1u;
  request.cfg.warmup_iterations = 0u;
  request.cfg.warmup_runs = 0u;

  std::string result_text;
  const process_capture capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(request),
      "process-seam-unknown-suite", result_text);

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
  const process_capture capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(request),
      "process-seam-conflicting-jsonl", result_text);

  CHECK(capture.exit_code == 2);
  CHECK(capture.stdout_text.empty());

  emel::bench::runner_result result = {};
  REQUIRE(emel::bench::parse_runner_result(result_text, result));
  CHECK(result.exit_code == 2);
  CHECK(result.error_kind == "invalid_request");
  CHECK(result.error_message.find("jsonl") != std::string::npos);
}

TEST_CASE(
    "bench runner process seam rejects incompatible jsonl suite requests") {
  emel::bench::runner_request generation_request = {};
  generation_request.mode = emel::bench::runner_mode::emel;
  generation_request.suite = "batch_planner";
  generation_request.cfg.iterations = 1u;
  generation_request.cfg.runs = 1u;
  generation_request.cfg.warmup_iterations = 0u;
  generation_request.cfg.warmup_runs = 0u;
  generation_request.generation_jsonl = true;

  std::string generation_result_text;
  const process_capture generation_capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(generation_request),
      "process-seam-bad-generation-jsonl-suite", generation_result_text);

  CHECK(generation_capture.exit_code == 2);
  CHECK(generation_capture.stdout_text.empty());

  emel::bench::runner_result generation_result = {};
  REQUIRE(emel::bench::parse_runner_result(generation_result_text,
                                           generation_result));
  CHECK(generation_result.exit_code == 2);
  CHECK(generation_result.error_kind == "invalid_request");
  CHECK(generation_result.error_message.find("generation jsonl") !=
        std::string::npos);

  emel::bench::runner_request diarization_request = {};
  diarization_request.mode = emel::bench::runner_mode::reference;
  diarization_request.suite = "generation";
  diarization_request.cfg.iterations = 1u;
  diarization_request.cfg.runs = 1u;
  diarization_request.cfg.warmup_iterations = 0u;
  diarization_request.cfg.warmup_runs = 0u;
  diarization_request.diarization_jsonl = true;

  std::string diarization_result_text;
  const process_capture diarization_capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(diarization_request),
      "process-seam-bad-diarization-jsonl-suite", diarization_result_text);

  CHECK(diarization_capture.exit_code == 2);
  CHECK(diarization_capture.stdout_text.empty());

  emel::bench::runner_result diarization_result = {};
  REQUIRE(emel::bench::parse_runner_result(diarization_result_text,
                                           diarization_result));
  CHECK(diarization_result.exit_code == 2);
  CHECK(diarization_result.error_kind == "invalid_request");
  CHECK(diarization_result.error_message.find("diarization jsonl") !=
        std::string::npos);
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
  const process_capture zero_runs_capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(zero_runs_request),
      "process-seam-zero-runs", zero_runs_result_text);

  CHECK(zero_runs_capture.exit_code == 2);
  CHECK(zero_runs_capture.stdout_text.empty());

  emel::bench::runner_result zero_runs_result = {};
  REQUIRE(emel::bench::parse_runner_result(zero_runs_result_text,
                                           zero_runs_result));
  CHECK(zero_runs_result.exit_code == 2);
  CHECK(zero_runs_result.error_kind == "invalid_request");
  CHECK(zero_runs_result.error_message.find("runs") != std::string::npos);

  emel::bench::runner_request too_many_runs_request = zero_runs_request;
  too_many_runs_request.cfg.runs = 26u;

  std::string too_many_runs_result_text;
  const process_capture too_many_runs_capture = run_serialized_request_capture(
      emel::bench::serialize_runner_request(too_many_runs_request),
      "process-seam-too-many-runs", too_many_runs_result_text);

  CHECK(too_many_runs_capture.exit_code == 2);
  CHECK(too_many_runs_capture.stdout_text.empty());

  emel::bench::runner_result too_many_runs_result = {};
  REQUIRE(emel::bench::parse_runner_result(too_many_runs_result_text,
                                           too_many_runs_result));
  CHECK(too_many_runs_result.exit_code == 2);
  CHECK(too_many_runs_result.error_kind == "invalid_request");
  CHECK(too_many_runs_result.error_message.find("runs") != std::string::npos);

  emel::bench::runner_request too_many_warmups_request = zero_runs_request;
  too_many_warmups_request.cfg.runs = 1u;
  too_many_warmups_request.cfg.warmup_runs = 26u;

  std::string too_many_warmups_result_text;
  const process_capture too_many_warmups_capture =
      run_serialized_request_capture(
          emel::bench::serialize_runner_request(too_many_warmups_request),
          "process-seam-too-many-warmup-runs", too_many_warmups_result_text);

  CHECK(too_many_warmups_capture.exit_code == 2);
  CHECK(too_many_warmups_capture.stdout_text.empty());

  emel::bench::runner_result too_many_warmups_result = {};
  REQUIRE(emel::bench::parse_runner_result(too_many_warmups_result_text,
                                           too_many_warmups_result));
  CHECK(too_many_warmups_result.exit_code == 2);
  CHECK(too_many_warmups_result.error_kind == "invalid_request");
  CHECK(too_many_warmups_result.error_message.find("warmup_runs") !=
        std::string::npos);
}

TEST_CASE(
    "benchmark runner registration is localized outside the orchestrator") {
  CHECK(emel::bench::registered_runner_count() >= 29u);
  CHECK(emel::bench::find_registered_runner("generation") != nullptr);
  CHECK(emel::bench::find_registered_runner("diarization_sortformer") !=
        nullptr);
  CHECK(emel::bench::find_registered_runner("speech_lm_moshi") != nullptr);
  CHECK(emel::bench::find_registered_runner("sm_scheduler") != nullptr);
  CHECK(emel::bench::find_registered_runner("tokenizer") != nullptr);
  CHECK(emel::bench::find_registered_runner("missing_suite") == nullptr);

  bool saw_generation = false;
  bool saw_speech_lm_moshi = false;
  bool saw_sm_scheduler = false;
  bool saw_tokenizer = false;
  for (std::size_t i = 0; i < emel::bench::registered_runner_count(); ++i) {
    saw_generation = saw_generation || emel::bench::registered_runner_suite_at(
                                           i) == std::string_view{"generation"};
    saw_speech_lm_moshi =
        saw_speech_lm_moshi || emel::bench::registered_runner_suite_at(i) ==
                                   std::string_view{"speech_lm_moshi"};
    saw_sm_scheduler =
        saw_sm_scheduler || emel::bench::registered_runner_suite_at(i) ==
                                std::string_view{"sm_scheduler"};
    saw_tokenizer = saw_tokenizer || emel::bench::registered_runner_suite_at(
                                         i) == std::string_view{"tokenizer"};
  }
  CHECK(saw_generation);
  CHECK(saw_speech_lm_moshi);
  CHECK(saw_sm_scheduler);
  CHECK(saw_tokenizer);
}

TEST_CASE(
    "bench runner orchestration no longer owns broad static registration") {
  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  const std::string registry_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner_registry.cpp");

  CHECK(runner_source.find("std::array<bench::test_case") == std::string::npos);
  CHECK(runner_source.find("append_emel_generation_cases") ==
        std::string::npos);
  CHECK(runner_source.find("append_reference_generation_cases") ==
        std::string::npos);
  CHECK(runner_source.find("bench::default_runner_cases()") !=
        std::string::npos);
  CHECK(registry_source.find("append_emel_generation_cases") !=
        std::string::npos);
  CHECK(registry_source.find("append_reference_generation_cases") !=
        std::string::npos);
}

TEST_CASE("bench runner suites build through independent object targets") {
  const std::string cmake_source =
      read_file(repo_root() / "tools" / "bench" / "CMakeLists.txt");

  CHECK(cmake_source.find("bench_runner_suite_${suite_name}") !=
        std::string::npos);
  CHECK(cmake_source.find("add_library(${target_name} OBJECT") !=
        std::string::npos);
  CHECK(cmake_source.find("$<TARGET_OBJECTS:${target_name}>") !=
        std::string::npos);
  CHECK(cmake_source.find(
            "configure_bench_runner_common_target(${target_name})") !=
        std::string::npos);
  CHECK(cmake_source.find(
            "configure_bench_runner_artifact_definitions(${target_name})") !=
        std::string::npos);
  CHECK(cmake_source.find(
            "add_bench_runner_suite(generation generation_bench.cpp") !=
        std::string::npos);
  CHECK(cmake_source.find("add_bench_runner_suite(diarization_sortformer") !=
        std::string::npos);
  CHECK(cmake_source.find("add_bench_runner_suite(speech_lm_moshi") !=
        std::string::npos);
  CHECK(cmake_source.find("add_bench_runner_suite(sm_scheduler") !=
        std::string::npos);
  CHECK(cmake_source.find("EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_kv\"") !=
        std::string::npos);
  CHECK(cmake_source.find(
            "EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_recurrent\"") !=
        std::string::npos);
  CHECK(
      cmake_source.find("EMEL_BENCH_SUITE_FILTER STREQUAL \"memory_hybrid\"") !=
      std::string::npos);
  CHECK(cmake_source.find("BENCH_RUNNER_SUITE_TARGETS") != std::string::npos);
}

TEST_CASE("bench runner emits internal sm scheduler cases") {
  const process_capture capture = run_suite_bench_capture(
      "sm_scheduler", "compare", "sm-scheduler-compare", true);

  CHECK(capture.exit_code == 0);
  CHECK(capture.stderr_text.find("error:") == std::string::npos);
  CHECK(capture.stdout_text.find("sm_scheduler/idle_async") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("sm_scheduler/busy_worker_async") !=
        std::string::npos);
  CHECK(capture.stdout_text.find("thread_pool") != std::string::npos);
  CHECK(capture.stdout_text.find("inline_co_sm") != std::string::npos);
}

TEST_CASE("bench runner rejects internal sm scheduler suite without explicit "
          "enable") {
  const process_capture capture = run_suite_bench_capture(
      "sm_scheduler", "compare", "sm-scheduler-disabled");

  CHECK(capture.exit_code != 0);
  CHECK(capture.stderr_text.find(
            "no benchmark entries matched selected suite 'sm_scheduler'") !=
        std::string::npos);
}

TEST_CASE(
    "benchmark dependency manifest covers registered runners conservatively") {
  namespace manifest = emel::bench::dependency_manifest;

  CHECK(manifest::kind_name(manifest::dependency_kind::source) == "source");
  CHECK(manifest::kind_name(manifest::dependency_kind::config) == "config");
  CHECK_FALSE(manifest::requires_full_gate({}));
  CHECK(manifest::requires_full_gate({.missing = true}));
  CHECK(manifest::requires_full_gate({.stale = true}));
  CHECK(manifest::requires_full_gate({.uncertain = true}));

  const auto all_records = manifest::records_for("all");
  REQUIRE_FALSE(all_records.empty());
  const std::size_t total_all_records = static_cast<std::size_t>(
      std::count_if(manifest::records().begin(), manifest::records().end(),
                    [](const auto &record) {
                      return record.runner == std::string_view{"all"};
                    }));
  CHECK(all_records.size() == total_all_records);

  bool saw_cmake = false;
  bool saw_quality_gate = false;
  for (const auto &record : all_records) {
    saw_cmake = saw_cmake ||
                record.path == std::string_view{"tools/bench/CMakeLists.txt"};
    saw_quality_gate =
        saw_quality_gate ||
        record.path == std::string_view{"scripts/quality_gates.sh"};
  }
  CHECK(saw_cmake);
  CHECK(saw_quality_gate);

  for (std::size_t i = 0; i < emel::bench::registered_runner_count(); ++i) {
    const std::string_view runner = emel::bench::registered_runner_suite_at(i);
    const auto records = manifest::records_for(runner);
    CHECK_MESSAGE(!records.empty(),
                  "missing manifest records for runner " << runner);
    const std::size_t total_records = static_cast<std::size_t>(std::count_if(
        manifest::records().begin(), manifest::records().end(),
        [runner](const auto &record) { return record.runner == runner; }));
    CHECK_MESSAGE(records.size() == total_records,
                  "manifest records are not contiguous for runner " << runner);
    bool has_source = false;
    for (const auto &record : records) {
      has_source =
          has_source || record.kind == manifest::dependency_kind::source;
    }
    CHECK_MESSAGE(has_source, "runner lacks source record " << runner);
  }

  const auto generation_records = manifest::records_for("generation");
  bool has_generation_config = false;
  bool has_generation_model = false;
  bool has_generation_script = false;
  for (const auto &record : generation_records) {
    has_generation_config = has_generation_config ||
                            record.kind == manifest::dependency_kind::config;
    has_generation_model =
        has_generation_model || record.kind == manifest::dependency_kind::model;
    has_generation_script = has_generation_script ||
                            record.kind == manifest::dependency_kind::script;
  }
  CHECK(has_generation_config);
  CHECK(has_generation_model);
  CHECK(has_generation_script);

  const auto needle_records = manifest::records_for("needle_graph");
  bool has_live_reference_driver = false;
  bool has_request_fixture = false;
  bool has_compare_wrapper = false;
  for (const auto &record : needle_records) {
    has_live_reference_driver =
        has_live_reference_driver ||
        record.path == "tools/bench/model/needle/cactus_reference.py";
    has_request_fixture =
        has_request_fixture ||
        record.path == "tests/fixtures/cact/needle-heldout-prompts.tsv";
    has_compare_wrapper =
        has_compare_wrapper || record.path == "scripts/bench.sh";
  }
  CHECK(has_live_reference_driver);
  CHECK(has_request_fixture);
  CHECK(has_compare_wrapper);

  const auto speech_lm_records = manifest::records_for("speech_lm_moshi");
  bool has_speech_lm_script = false;
  bool has_speech_lm_setup = false;
  bool has_speech_lm_moshi_binding = false;
  bool has_personaplex_emel_runner = false;
  for (const auto &record : speech_lm_records) {
    has_speech_lm_script = has_speech_lm_script ||
                           record.path == "scripts/bench_moshi_lm_compare.sh";
    has_speech_lm_setup = has_speech_lm_setup ||
                          record.path == "scripts/setup_moshi_cpp_reference.sh";
    has_speech_lm_moshi_binding =
        has_speech_lm_moshi_binding ||
        record.path == std::string_view{"src/emel/model/moshi"};
    has_personaplex_emel_runner =
        has_personaplex_emel_runner ||
        record.path ==
            std::string_view{"tools/bench/speech/personaplex_emel_runner.cpp"};
  }
  CHECK(has_speech_lm_script);
  CHECK(has_speech_lm_setup);
  CHECK(has_speech_lm_moshi_binding);
  CHECK(has_personaplex_emel_runner);
  CHECK(manifest::records_for("missing_runner").empty());
}

TEST_CASE("needle combined snapshot compare filters only live diagnostics") {
  const std::string wrapper = read_file(repo_root() / "scripts" / "bench.sh");
  CHECK(wrapper.find(
            "if [[ \"$SUITE_FILTER\" == \"needle_graph\" ]]; then\n"
            "    EMEL_BENCH_NEEDLE_REQUEST_COMPARE=1 run_bench_runner \"$build_dir\" --mode=emel > \"$snapshot_output\"\n"
            "    run_needle_graph_compare \"$build_dir\" > \"$compare_output\"") !=
        std::string::npos);

#if !defined(_WIN32)
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "emel-bench-runner-tests" /
      "needle-snapshot-filter";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path snapshot_path = tmp_dir / "snapshot.txt";
  const std::filesystem::path current_path = tmp_dir / "current.txt";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  const std::string wrapper_command =
      "EMEL_BENCH_TEST_SNAPSHOT_OUTPUT=" +
      quote_arg_posix(snapshot_path.string()) +
      " EMEL_BENCH_TEST_CURRENT_SNAPSHOT=" +
      quote_arg_posix(current_path.string()) +
      " EMEL_BENCH_TEST_SUITE_FILTER=needle_graph "
      "EMEL_BENCH_TEST_SNAPSHOT_FILTER=1 " +
      quote_arg_posix((repo_root() / "scripts" / "bench.sh").string()) +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());

  write_file(snapshot_path,
             "# needle_graph: lane=emel case=needle/graph/request "
             "reference=live_cactus_native comparable=false\n"
             "needle/graph/request ns_per_op=100 tokens_per_second=10\n");
  process_capture capture = run_command_capture(
      wrapper_command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text == "live-diagnostics-only\n");
  CHECK(read_file(current_path).empty());

  write_file(snapshot_path,
             "# needle_graph: lane=emel case=needle/graph/request "
             "reference=live_cactus_native comparable=false\n"
             "needle/graph/request ns_per_op=100 tokens_per_second=10\n"
             "# ordinary_suite: lane=emel case=ordinary/new_case\n"
             "ordinary/new_case ns_per_op=200\n");
  capture = run_command_capture(wrapper_command, stdout_path, stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text == "baseline-required\n");
  CHECK(read_file(current_path) == "ordinary/new_case ns_per_op=200\n");

  const std::string global_wrapper_command =
      "EMEL_BENCH_TEST_SNAPSHOT_OUTPUT=" +
      quote_arg_posix(snapshot_path.string()) +
      " EMEL_BENCH_TEST_CURRENT_SNAPSHOT=" +
      quote_arg_posix(current_path.string()) +
      " EMEL_BENCH_TEST_SNAPSHOT_FILTER=1 " +
      quote_arg_posix((repo_root() / "scripts" / "bench.sh").string()) +
      " > " + quote_arg_posix(stdout_path.string()) + " 2> " +
      quote_arg_posix(stderr_path.string());
  capture = run_command_capture(global_wrapper_command, stdout_path,
                                stderr_path);
  CHECK(capture.exit_code == 0);
  CHECK(capture.stdout_text == "baseline-required\n");
  CHECK(read_file(current_path).find("needle/graph/request ns_per_op=100") !=
        std::string::npos);
  CHECK(read_file(current_path).find("ordinary/new_case ns_per_op=200") !=
        std::string::npos);

  const std::string compare_gate =
      read_file(repo_root() / "scripts" / "bench_compare_gate.awk");
  CHECK(compare_gate.find("new benchmark entry without baseline") !=
        std::string::npos);
  CHECK(compare_gate.find("live_cactus_native") == std::string::npos);

  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  CHECK_FALSE(ec);
#endif
}

TEST_CASE(
    "benchmark dependency manifest renders and writes deterministic output") {
  namespace manifest = emel::bench::dependency_manifest;

  const std::string rendered = manifest::render();
  CHECK(rendered == manifest::render());
  CHECK(rendered.rfind(std::string(manifest::k_schema) + "\n", 0u) == 0u);
  CHECK(rendered.find("full_gate_on=missing,stale,uncertain\n") !=
        std::string::npos);
  CHECK(rendered.find("record runner=generation kind=source "
                      "path=tools/bench/generation_bench.cpp") !=
        std::string::npos);
  CHECK(rendered.find("record runner=diarization_sortformer kind=source "
                      "path=tools/bench/diarization/sortformer_bench.cpp") !=
        std::string::npos);
  CHECK(rendered.find("record runner=speech_lm_moshi kind=source "
                      "path=tools/bench/speech/lm_moshi_bench.cpp") !=
        std::string::npos);
  CHECK(rendered.find("record runner=speech_lm_moshi kind=source "
                      "path=tools/bench/speech/personaplex_emel_runner.cpp") !=
        std::string::npos);

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

TEST_CASE("speech Moshi attention benchmark owns per-layer KV cache slices") {
  const std::string source = read_file(repo_root() / "tools" / "bench" /
                                       "speech" / "lm_moshi_bench.cpp");

  CHECK(
      source.find(
          "key_cache(static_cast<std::size_t>(model.moshi_lm.num_layers) *") !=
      std::string::npos);
  CHECK(source.find("layer_offsets[index] = index * per_layer_cache;") !=
        std::string::npos);
}

#if !defined(_WIN32)
TEST_CASE("moshi lm wrapper gives --model priority over inherited env") {
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() /
                                        "emel-bench-runner-tests" /
                                        "moshi-lm-model-priority";
  const std::filesystem::path fake_bin_dir = tmp_dir / "bin";
  const std::filesystem::path build_dir = tmp_dir / "build";
  const std::filesystem::path fake_runner = build_dir / "bench_runner";
  const std::filesystem::path cli_model = tmp_dir / "cli-model.gguf";
  const std::filesystem::path stale_model = tmp_dir / "stale-model.gguf";
  const std::filesystem::path invoked_path = tmp_dir / "runner-invoked.txt";
  const std::filesystem::path stdout_path = tmp_dir / "stdout.txt";
  const std::filesystem::path stderr_path = tmp_dir / "stderr.txt";
  std::filesystem::create_directories(fake_bin_dir);
  std::filesystem::create_directories(build_dir);
  write_file(cli_model, "cli");
  write_file(stale_model, "stale");

  for (const char *tool : {"cmake", "ninja", "git"}) {
    const std::filesystem::path tool_path = fake_bin_dir / tool;
    write_file(tool_path, "#!/bin/sh\nexit 0\n");
    make_executable(tool_path);
  }
  const std::filesystem::path xcrun_path = fake_bin_dir / "xcrun";
  write_file(xcrun_path, "#!/bin/sh\nexit 99\n");
  make_executable(xcrun_path);
  write_file(fake_runner,
             "#!/bin/sh\n"
             "printf 'personaplex=%s\\nmoshi=%s\\nbench=%s\\nargs=%s\\n' "
             "\"$EMEL_PERSONAPLEX_LM_MODEL\" \"$EMEL_MOSHI_LM_MODEL\" "
             "\"$EMEL_BENCH_SPEECH_LM_MOSHI_MODEL\" \"$*\" "
             "> \"$EMEL_TEST_INVOKED_PATH\"\n"
             "exit 0\n");
  make_executable(fake_runner);

  std::string command =
      "PATH=" + quote_arg_posix(fake_bin_dir.string()) + ":$PATH ";
  command +=
      "EMEL_TEST_INVOKED_PATH=" + quote_arg_posix(invoked_path.string()) + " ";
  command +=
      "EMEL_MOSHI_LM_COMPARE_BUILD_DIR=" + quote_arg_posix(build_dir.string()) +
      " ";
  command +=
      "EMEL_PERSONAPLEX_LM_MODEL=" + quote_arg_posix(stale_model.string()) +
      " ";
  command += quote_arg_posix("/bin/bash") + " " +
             quote_arg_posix(bench_moshi_lm_compare_wrapper_path().string());
  command +=
      " --run-only --system --model " + quote_arg_posix(cli_model.string());
  command += " > " + quote_arg_posix(stdout_path.string());
  command += " 2> " + quote_arg_posix(stderr_path.string());

  const process_capture capture =
      run_command_capture(command, stdout_path, stderr_path);

  const std::string invoked = read_file(invoked_path);
  CHECK_MESSAGE(!invoked.empty(), capture.stderr_text);
  CHECK(invoked.find("personaplex=" + cli_model.string()) != std::string::npos);
  CHECK(invoked.find("moshi=" + cli_model.string()) != std::string::npos);
  CHECK(invoked.find("bench=" + cli_model.string()) != std::string::npos);
  CHECK(invoked.find("personaplex=" + stale_model.string()) ==
        std::string::npos);
}

TEST_CASE("moshi lm wrapper keeps build-only model-free") {
  const std::string script = read_file(bench_moshi_lm_compare_wrapper_path());
  const std::size_t mutual_exclusion =
      script.find("if $BUILD_ONLY && $RUN_ONLY; then");
  const std::size_t model_guard = script.find("if ! $BUILD_ONLY; then");
  const std::size_t setup_call = script.find(
      "setup_output=\"$(\"$ROOT_DIR/scripts/setup_moshi_cpp_reference.sh\")\"");
  const std::size_t tool_check = script.find("for tool in cmake ninja git");
  const std::size_t build_only_exit = script.find("if $BUILD_ONLY; then");

  REQUIRE(mutual_exclusion != std::string::npos);
  REQUIRE(model_guard != std::string::npos);
  REQUIRE(setup_call != std::string::npos);
  REQUIRE(tool_check != std::string::npos);
  REQUIRE(build_only_exit != std::string::npos);
  CHECK(mutual_exclusion < model_guard);
  CHECK(model_guard < setup_call);
  CHECK(setup_call < tool_check);
  CHECK(tool_check < build_only_exit);
}

TEST_CASE("moshi lm wrapper defers Zig setup outside system run-only") {
  const std::string script = read_file(bench_moshi_lm_compare_wrapper_path());
  const std::size_t source_zig =
      script.find("source \"$ROOT_DIR/scripts/zig_toolchain.sh\"");
  const std::size_t build_guard =
      script.find("if ! $RUN_ONLY && $USE_ZIG; then");

  REQUIRE(source_zig != std::string::npos);
  REQUIRE(build_guard != std::string::npos);
  CHECK(build_guard < source_zig);
}

TEST_CASE("moshi lm wrapper and runner search common companion layouts") {
  const std::string script = read_file(bench_moshi_lm_compare_wrapper_path());
  CHECK(script.find("$ROOT_DIR/../companion/zig-out/"
                    "personaplex-emel-converted") != std::string::npos);
  CHECK(script.find("$ROOT_DIR/../../companion/zig-out/"
                    "personaplex-emel-converted") != std::string::npos);
  CHECK(script.find("$ROOT_DIR/../../../companion/zig-out/"
                    "personaplex-emel-converted") != std::string::npos);

  const std::string runner = read_file(repo_root() / "tools" / "bench" /
                                       "speech" / "lm_moshi_bench.cpp");
  CHECK(runner.find("root.parent_path() / \"companion\"") != std::string::npos);
  CHECK(runner.find("root.parent_path().parent_path() / \"companion\"") !=
        std::string::npos);
  CHECK(runner.find("root.parent_path().parent_path().parent_path() / "
                    "\"companion\"") != std::string::npos);
}
#endif

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
      {"--check-dependency-manifest", manifest_path.string(),
       "--dependency-manifest-uncertain"},
      "bench-manifest-uncertain");
  CHECK(uncertain_capture.exit_code == 3);
  CHECK(uncertain_capture.stdout_text.find("full_gate=1") != std::string::npos);
  CHECK(uncertain_capture.stdout_text.find("reason=uncertain") !=
        std::string::npos);

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
  CHECK(missing_capture.stdout_text.find("reason=missing") !=
        std::string::npos);

  process_capture invalid_capture = run_bench_runner_capture(
      {"--write-dependency-manifest", manifest_path.string(),
       "--dependency-manifest-uncertain"},
      "bench-manifest-invalid");
  CHECK(invalid_capture.exit_code == 2);
  CHECK(invalid_capture.stderr_text.find(
            "error: invalid dependency manifest arguments") !=
        std::string::npos);
}

TEST_CASE("shared benchmark orchestration stays lane-neutral and "
          "actor-boundary clean") {
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

  for (const std::filesystem::path &path : shared_paths) {
    const std::string raw_source = read_file(path);
    REQUIRE_MESSAGE(!raw_source.empty(), "missing source " << path.string());
    const std::string source = actor_boundary_scan_source(path, raw_source);
    for (const std::string &pattern : forbidden_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.string() << " contains forbidden pattern " << pattern);
    }
  }

  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  CHECK(runner_source.find("append_emel_generation_cases") ==
        std::string::npos);
  CHECK(runner_source.find("append_reference_generation_cases") ==
        std::string::npos);
  CHECK(runner_source.find("append_emel_sortformer_diarization_cases") ==
        std::string::npos);
  CHECK(runner_source.find("append_reference_sortformer_diarization_cases") ==
        std::string::npos);
}

TEST_CASE("maintained benchmark runner sources avoid actor internals") {
  const std::vector<std::string> forbidden_patterns = {
      "/actions.hpp",
      "/guards.hpp",
      "::action::",
      "::guard::",
      "emel/batch/planner/detail.hpp",
      "emel/diarization/sortformer/request/detail.hpp",
      "emel/diarization/sortformer/executor/detail.hpp",
      "emel/diarization/sortformer/pipeline/detail.hpp",
      "emel/text/generator/detail.hpp",
      "emel/text/generator/prefill/detail.hpp",
      "emel/text/jinja/formatter/detail.hpp",
      "emel/text/jinja/parser/detail.hpp",
      "emel::batch::planner::detail::",
      "emel::diarization::sortformer::request::detail::",
      "emel::diarization::sortformer::executor::detail::",
      "emel::diarization::sortformer::pipeline::detail::",
      "emel::text::generator::detail::",
      "emel::text::generator::prefill::detail::",
      "emel::text::jinja::formatter::detail::",
      "emel::text::jinja::parser::detail::",
  };

  std::size_t checked_files = 0u;
  const std::filesystem::path bench_dir = repo_root() / "tools" / "bench";
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(bench_dir)) {
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

    const std::string raw_source = read_file(path);
    REQUIRE_MESSAGE(!raw_source.empty(), "missing source " << path.string());
    const std::string source = actor_boundary_scan_source(path, raw_source);
    checked_files += 1u;
    for (const std::string &pattern : forbidden_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    path.string()
                        << " contains forbidden actor-internal pattern "
                        << pattern);
    }
  }
  CHECK(checked_files > 20u);
}

TEST_CASE("maintained benchmark behavior coverage remains source-backed") {
  const std::string tests_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner_tests.cpp");

  CHECK(
      tests_source.find("bench_main shim delegates to benchmark runner cli") !=
      std::string::npos);
  CHECK(tests_source.find("bench_runner generation jsonl emits "
                          "manifest-driven workload") != std::string::npos);
  CHECK(tests_source.find("bench_runner diarization jsonl emits structured "
                          "maintained parity") != std::string::npos);
  CHECK(tests_source.find("benchmark runner registration is localized outside "
                          "the orchestrator") != std::string::npos);
  CHECK(tests_source.find("benchmark dependency manifest renders and writes "
                          "deterministic output") != std::string::npos);
  CHECK(
      tests_source.find("shared benchmark orchestration stays lane-neutral") !=
      std::string::npos);
  CHECK(tests_source.find(
            "maintained benchmark runner sources avoid actor internals") !=
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
  for (const std::string &pattern : forbidden_source_patterns) {
    CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                  "generation_bench.cpp contains forbidden pattern "
                      << pattern);
  }

  const std::string marker = "bool measure_emel_stage_probe(";
  const auto start = source.find(marker);
  REQUIRE(start != std::string::npos);
  const auto end = source.find("bool measure_reference_stage_probe(", start);
  REQUIRE(end != std::string::npos);

  const std::string probe_source = source.substr(start, end - start);
  CHECK(probe_source.find("emel::text::generator::detail::") ==
        std::string::npos);
  CHECK(probe_source.find("emel::text::generator::guard::") ==
        std::string::npos);
  CHECK(probe_source.find("emel::text::generator::prefill::guard::") ==
        std::string::npos);
  CHECK(probe_source.find("emel::text::generator::action::context") ==
        std::string::npos);
  CHECK(probe_source.find("emel::text::generator::prefill::action::context") ==
        std::string::npos);
}

TEST_CASE("sortformer_diarization_bench_uses_public_actor_surfaces") {
  const std::array<std::filesystem::path, 2> source_paths = {
      repo_root() / "tools" / "bench" / "diarization" / "sortformer_bench.cpp",
      repo_root() / "tools" / "bench" / "diarization" /
          "sortformer_fixture.hpp",
  };
  const std::vector<std::string> forbidden_source_patterns = {
      "#include \"emel/diarization/sortformer/request/actions.hpp\"",
      "#include \"emel/diarization/sortformer/request/detail.hpp\"",
      "#include \"emel/diarization/sortformer/request/guards.hpp\"",
      "#include \"emel/diarization/sortformer/encoder/actions.hpp\"",
      "#include \"emel/diarization/sortformer/encoder/detail.hpp\"",
      "#include "
      "\"emel/diarization/sortformer/encoder/feature_extractor/detail.hpp\"",
      "#include \"emel/diarization/sortformer/encoder/guards.hpp\"",
      "#include \"emel/diarization/sortformer/executor/actions.hpp\"",
      "#include \"emel/diarization/sortformer/executor/detail.hpp\"",
      "#include \"emel/diarization/sortformer/executor/guards.hpp\"",
      "#include \"emel/diarization/sortformer/modules/detail.hpp\"",
      "#include \"emel/diarization/sortformer/output/actions.hpp\"",
      "#include \"emel/diarization/sortformer/output/detail.hpp\"",
      "#include \"emel/diarization/sortformer/output/guards.hpp\"",
      "#include \"emel/diarization/sortformer/pipeline/actions.hpp\"",
      "#include \"emel/diarization/sortformer/pipeline/detail.hpp\"",
      "#include \"emel/diarization/sortformer/pipeline/guards.hpp\"",
      "#include \"emel/model/sortformer/detail.hpp\"",
      "emel::diarization::sortformer::request::action::",
      "emel::diarization::sortformer::request::detail::",
      "emel::diarization::sortformer::request::guard::",
      "emel::diarization::sortformer::encoder::action::",
      "emel::diarization::sortformer::encoder::detail::",
      "emel::diarization::sortformer::encoder::feature_extractor::detail::",
      "emel::diarization::sortformer::encoder::guard::",
      "emel::diarization::sortformer::executor::action::",
      "emel::diarization::sortformer::executor::detail::",
      "emel::diarization::sortformer::executor::guard::",
      "emel::diarization::sortformer::modules::detail::",
      "emel::diarization::sortformer::output::action::",
      "emel::diarization::sortformer::output::detail::",
      "emel::diarization::sortformer::output::guard::",
      "emel::diarization::sortformer::pipeline::action::",
      "emel::diarization::sortformer::pipeline::detail::",
      "emel::diarization::sortformer::pipeline::guard::",
      "emel::model::sortformer::detail::",
  };

  for (const auto &source_path : source_paths) {
    const std::string source = read_file(source_path);
    REQUIRE_MESSAGE(!source.empty(), "missing source " << source_path.string());
    for (const std::string &pattern : forbidden_source_patterns) {
      CHECK_MESSAGE(source.find(pattern) == std::string::npos,
                    source_path.string()
                        << " bypasses Sortformer actor boundary with "
                        << pattern);
    }
  }

  const std::array<std::filesystem::path, 5> public_facade_paths = {
      repo_root() / "src" / "emel" / "model" / "sortformer" / "any.hpp",
      repo_root() / "src" / "emel" / "diarization" / "sortformer" / "request" /
          "events.hpp",
      repo_root() / "src" / "emel" / "diarization" / "sortformer" / "output" /
          "any.hpp",
      repo_root() / "src" / "emel" / "diarization" / "sortformer" / "pipeline" /
          "any.hpp",
      repo_root() / "src" / "emel" / "diarization" / "sortformer" / "executor" /
          "events.hpp",
  };
  for (const auto &facade_path : public_facade_paths) {
    const std::string source = read_file(facade_path);
    REQUIRE_MESSAGE(!source.empty(), "missing source " << facade_path.string());
    CHECK_MESSAGE(source.find("/detail.hpp") == std::string::npos,
                  facade_path.string() << " includes a detail header");
    CHECK_MESSAGE(source.find("::detail::") == std::string::npos,
                  facade_path.string() << " exposes a detail namespace");
  }

  CHECK_FALSE(std::filesystem::exists(repo_root() / "src" / "emel" /
                                      "diarization" / "request"));
  CHECK_FALSE(std::filesystem::exists(repo_root() / "tests" / "diarization" /
                                      "request"));
  CHECK(source_paths[0].string().find("sortformer_bench.cpp") !=
        std::string::npos);
  const std::string bench_source = read_file(source_paths[0]);
  CHECK(bench_source.find("EMEL_DIARIZATION_STAGE_PROFILE") ==
        std::string::npos);
  CHECK(bench_source.find("stage_profile_") == std::string::npos);
}

TEST_CASE("generation_stage_probe_selector_is_explicit") {
  const std::string generation_source =
      read_file(repo_root() / "tools" / "bench" / "generation_bench.cpp");
  const std::string runner_source =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  REQUIRE_FALSE(generation_source.empty());
  REQUIRE_FALSE(runner_source.empty());

  CHECK(generation_source.find("EMEL_GENERATION_STAGE_PROBE") !=
        std::string::npos);
  CHECK(generation_source.find("generation_stage_probe_selection::selected") !=
        std::string::npos);
  CHECK(generation_source.find(
            "should_capture_generation_stage_probe(*spec, generation_case)") !=
        std::string::npos);
  CHECK(runner_source.find("print_generation_stage_probes();") !=
        std::string::npos);
}

TEST_CASE("bench_runner generation jsonl emits manifest-driven workload "
          "metadata and explicit comparability") {
  const process_capture emel_capture =
      run_generation_bench_capture("emel", true);
  CHECK(emel_capture.exit_code == 0);
  CHECK(emel_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"schema\":\"generation_compare/v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"emel\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"reference\"") ==
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"backend_id\":\"emel.generator\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"benchmark_lane\":\"single\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"benchmark_lane\":\"multithreaded\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"thread_count\":1") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"thread_count\":8") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"thread_contract\":\"emel_serial_matmul_lanes=1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"thread_contract\":\"emel_parallel_matmul_lanes=8\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"backend_id\":\"cpp.reference.llama_cpp\"") == std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_id\":\"" +
            std::string{k_bounded_generation_workload_id} + "\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_manifest_path\":\"tools/bench/generation_variants/"
            "lfm2/single_user_hello/parity/max_tokens_1.json\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"prompt_fixture_id\":\"single_user_hello_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"prompt_fixture_path\":\"tools/bench/generation_prompts/"
            "single_user_hello.json\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"prompt_id\":\"single_user:hello\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"formatter_mode\":\"chat_template_supported_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"sampling_id\":\"argmax_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparable\":true") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"tokens_per_second\":") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"kernel_dispatch_calls\":") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"flash_attention_dispatch_calls\":") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"native_quantized_stage_count\":") !=
        std::string::npos);
  const std::filesystem::path gemma4_fixture_path =
      repo_root() / "tests" / "models" / "gemma-4-e2b-it-Q8_0.gguf";
  if (std::filesystem::exists(gemma4_fixture_path)) {
    CHECK(emel_capture.stdout_text.find(
              "\"comparison_mode\":\"single_lane\"") != std::string::npos);
    CHECK(emel_capture.stdout_text.find("\"note\":\"reference_lane_unavailable_"
                                        "for_maintained_compare_surface\"") !=
          std::string::npos);
  }
  CHECK(emel_capture.stdout_text.find("\"output_path\":\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"output_token_ids_path\":\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("ns/op,") == std::string::npos);

  const process_capture reference_capture =
      run_generation_bench_capture("reference", true);
  CHECK(reference_capture.exit_code == 0);
  CHECK(reference_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"schema\":\"generation_compare/v1\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"reference\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"emel\"") ==
        std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"backend_id\":\"cpp.reference.llama_cpp\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"benchmark_lane\":\"single\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"benchmark_lane\":\"multithreaded\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"thread_count\":1") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"thread_count\":8") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"thread_contract\":\"llama.cpp_n_threads=") != std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"backend_id\":\"emel.generator\"") == std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"workload_manifest_path\":\"tools/bench/generation_variants/") !=
        std::string::npos);
  const bool saw_supported_reference_formatter =
      reference_capture.stdout_text.find(
          "\"formatter_mode\":\"chat_template_supported_qwen_v1\"") !=
          std::string::npos ||
      reference_capture.stdout_text.find(
          "\"formatter_mode\":\"chat_template_supported_v1\"") !=
          std::string::npos;
  CHECK(saw_supported_reference_formatter);
  CHECK(reference_capture.stdout_text.find("\"comparable\":true") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"tokens_per_second\":") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"kernel_dispatch_calls\":0") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"comparison_mode\":\"single_lane\"") == std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"formatter_contract\":\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"output_path\":\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"output_token_ids_path\":\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("ns/op,") == std::string::npos);
}

TEST_CASE("bench_runner diarization jsonl emits structured maintained parity "
          "metadata") {
  const process_capture emel_capture =
      run_diarization_bench_capture("emel", true);
  CHECK(emel_capture.exit_code == 0);
  CHECK(emel_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"schema\":\"diarization_compare/v1\"") != std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"emel\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"lane\":\"reference\"") ==
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"backend_id\":\"emel.diarization.sortformer\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparison_mode\":\"parity\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"model_id\":\"diar_streaming_sortformer_4spk_v2_1_gguf\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"fixture_id\":\"ami_en2002b_mix_"
                                      "headset_137.00_152.04_16khz_mono\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find(
            "\"workload_id\":\"diarization_sortformer_pipeline_v1\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"comparable\":true") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("\"output_path\":\"") !=
        std::string::npos);
  CHECK(emel_capture.stdout_text.find("ns/op,") == std::string::npos);

  const process_capture reference_capture =
      run_diarization_bench_capture("reference", true);
  CHECK(reference_capture.exit_code == 0);
  CHECK(reference_capture.stderr_text.find("error:") == std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"schema\":\"diarization_compare/v1\"") != std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"reference\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"lane\":\"emel\"") ==
        std::string::npos);
  CHECK(reference_capture.stdout_text.find(
            "\"backend_id\":\"recorded.diarization.baseline\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"comparison_mode\":\"parity\"") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("\"comparable\":true") !=
        std::string::npos);
  CHECK(reference_capture.stdout_text.find("ns/op,") == std::string::npos);
}

TEST_CASE("generation prompt fixture parser ignores quoted key names inside "
          "text values") {
  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() /
                                        "emel-bench-runner-tests" /
                                        "prompt-key-text";
  const std::filesystem::path prompt_path = tmp_dir / "prompt.json";
  std::error_code ec = {};
  std::filesystem::remove_all(tmp_dir, ec);
  std::filesystem::create_directories(tmp_dir);

  std::ofstream output(prompt_path);
  REQUIRE(output.good());
  output
      << "{\n"
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
  CHECK(emel::bench::load_generation_prompt_fixture(prompt_path, fixture,
                                                    &error));
  CHECK(error.empty());
  CHECK(fixture.text == "literal marker \"prompt_id\" before metadata");
  CHECK(fixture.prompt_id == "single_user:quoted_key");
}

TEST_CASE("generation workload manifests are discovered deterministically") {
  std::vector<emel::bench::generation_workload_manifest> manifests = {};
  std::string error = {};
  CHECK(emel::bench::load_generation_workload_manifests(repo_root(), manifests,
                                                        &error));
  CHECK(error.empty());
  CHECK(manifests.size() >= 13u);
  REQUIRE(!manifests.empty());
  CHECK(manifests.front().workload_manifest_path.find(
            "tools/bench/generation_variants/") == 0u);
  CHECK(manifests.front().workload_manifest_path !=
        "tools/bench/generation_variants/" +
            std::filesystem::path(manifests.front().workload_manifest_path)
                .filename()
                .string());
  CHECK(
      std::any_of(manifests.begin(), manifests.end(), [](const auto &manifest) {
        return manifest.id == "qwen3_single_user_hello_max_tokens_1_v1";
      }));
}
