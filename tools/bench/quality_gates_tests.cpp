#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

namespace {

std::filesystem::path repo_root() {
#ifdef BENCH_REPO_ROOT
  return BENCH_REPO_ROOT;
#else
  return std::filesystem::current_path();
#endif
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

#if !defined(_WIN32)
void write_file(const std::filesystem::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  REQUIRE(output.good());
  output << text;
  REQUIRE(output.good());
}

std::string shell_quote(const std::string &text) {
  std::string quoted{"'"};
  for (const char ch : text) {
    if (ch == '\'') quoted += "'\\''";
    else quoted.push_back(ch);
  }
  quoted.push_back('\'');
  return quoted;
}

struct command_result {
  int status = 1;
  std::string output;
};

command_result run_command(const std::string &command,
                           const std::filesystem::path &output_path) {
  const std::string full_command =
      command + " >" + shell_quote(output_path.string()) + " 2>&1";
  const int raw = std::system(full_command.c_str());
  command_result result{};
  result.status = raw == 0 ? 0 : 1;
  result.output = read_file(output_path);
  std::error_code ec;
  std::filesystem::remove(output_path, ec);
  return result;
}

// Run the extracted compare gate (scripts/bench_compare_gate.awk) against the
// given baseline and current snapshots, returning the process exit status
// (0 => gate passed, non-zero => gate failed).
int run_compare_gate(const std::string &baseline, const std::string &current,
                     const std::string &host_arch) {
  static int fixture_counter = 0;
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("bench_gate_test_" + std::to_string(++fixture_counter) + "_" +
       std::to_string(std::rand()));
  std::filesystem::create_directories(dir);
  const std::filesystem::path baseline_path = dir / "baseline.txt";
  const std::filesystem::path current_path = dir / "current.txt";
  write_file(baseline_path, baseline);
  write_file(current_path, current);

  const std::filesystem::path gate =
      repo_root() / "scripts" / "bench_compare_gate.awk";
  const std::string command =
      "awk -v tol=0.30 -v abs_tol=5000 -v strict_regression=0 -v scoped=0 "
      "-v host_arch=" +
      host_arch + " -f '" + gate.string() + "' '" + baseline_path.string() +
      "' '" + current_path.string() + "' >/dev/null 2>&1";
  const int raw = std::system(command.c_str());
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  // Collapse to a portable pass/fail: 0 => gate passed, anything else =>
  // failed. The exact non-zero encoding differs by platform/shell, so callers
  // only distinguish pass from fail.
  return raw == 0 ? 0 : 1;
}
#endif

} // namespace

TEST_CASE("quality gates full benchmark branch preserves failure status") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t full_branch_start = script.find("if $bench_full; then");
  REQUIRE(full_branch_start != std::string::npos);

  const std::size_t scoped_branch_end = script.find(
      "if [[ ${#bench_suites[@]} -eq 0 ]]; then", full_branch_start);
  REQUIRE(scoped_branch_end != std::string::npos);

  const std::string full_branch =
      script.substr(full_branch_start, scoped_branch_end - full_branch_start);

  CHECK(full_branch.find("run_step_allow_fail bench_snapshot") !=
        std::string::npos);
  CHECK(full_branch.find("EMEL_BENCH_INTERNAL=1") != std::string::npos);
  CHECK(full_branch.find("full_status=$?") != std::string::npos);
  CHECK(full_branch.find("run_full_benchmark_opt_in_suites") !=
        std::string::npos);
  CHECK(full_branch.find("if ! run_full_benchmark_opt_in_suites") ==
        std::string::npos);
  CHECK(full_branch.find("opt_in_status=$?") != std::string::npos);
  CHECK(full_branch.find("return \"$opt_in_status\"") != std::string::npos);
  CHECK(full_branch.find("return $?") == std::string::npos);
}

TEST_CASE("fuzz smoke validates the selected compiler's libFuzzer support") {
  const std::string script =
      read_file(repo_root() / "scripts" / "fuzz_smoke.sh");

  const std::size_t probe_start = script.find("check_libfuzzer_runtime()");
  REQUIRE(probe_start != std::string::npos);
  const std::size_t probe_call =
      script.find("check_libfuzzer_runtime\n", probe_start);
  REQUIRE(probe_call != std::string::npos);
  const std::size_t cmake_call = script.find("cmake -S", probe_call);
  REQUIRE(cmake_call != std::string::npos);

  const std::string probe =
      script.substr(probe_start, probe_call - probe_start);
  CHECK(probe.find("\"$fuzz_cxx\"") != std::string::npos);
  CHECK(probe.find("-fsanitize=fuzzer") != std::string::npos);
  CHECK(probe.find("cannot link a libFuzzer executable") !=
        std::string::npos);
  CHECK(probe_call < cmake_call);
  CHECK(script.find("lib/clang/*/lib/darwin/libclang_rt.fuzzer_osx.a") ==
        std::string::npos);
  CHECK(script.find("if [[ \"$fuzz_cc\" == \"clang\" ]]") ==
        std::string::npos);
  CHECK(script.find("if [[ \"$(uname -s)\" == \"Darwin\" ]]") !=
        std::string::npos);
  CHECK(script.find("/opt/homebrew/opt/llvm") != std::string::npos);
  CHECK(script.find("/usr/local/opt/llvm") != std::string::npos);
}

TEST_CASE("quality gates exclude nested sml machine headers from coverage "
          "source set") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start =
      script.find("is_coverage_excluded_src_file()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("changed_files=()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("src/emel/**/*/sm.hpp") != std::string::npos);
}

TEST_CASE("coverage domains use bounded doctest shards") {
  const std::string cmake = read_file(repo_root() / "CMakeLists.txt");
  CHECK(cmake.find("test_shard STREQUAL \"text_generator\"") !=
        std::string::npos);
  CHECK(cmake.find("test_source MATCHES \"^tests/text/generator/\"") !=
        std::string::npos);
  CHECK(cmake.find("test_shard STREQUAL \"embeddings\"") !=
        std::string::npos);
  CHECK(cmake.find("test_source MATCHES \"^tests/embeddings/\"") !=
        std::string::npos);
  CHECK(cmake.find("test_shard STREQUAL \"logits_and_token\"") !=
        std::string::npos);
  CHECK(cmake.find("test_source MATCHES \"^tests/(logits|token)/\"") !=
        std::string::npos);
  CHECK(cmake.find("\n    text_generator\n    \"*tests/text/generator/*\"") !=
        std::string::npos);
  CHECK(cmake.find("\n    embeddings\n    \"*tests/embeddings/*\"") !=
        std::string::npos);
  CHECK(cmake.find(
            "\n    logits_and_token\n    \"*tests/logits/*,*tests/token/*\"") !=
        std::string::npos);

  const std::string coverage =
      read_file(repo_root() / "scripts" / "test_with_coverage.sh");
  CHECK(coverage.find("text_generator)\n"
                      "      add_selected_test_dir tests/text/generator") !=
        std::string::npos);
  CHECK(coverage.find("embeddings)\n"
                      "      add_selected_test_dir tests/embeddings") !=
        std::string::npos);
  CHECK(coverage.find("logits_and_token)\n"
                      "      add_selected_test_dir tests/logits\n"
                      "      add_selected_test_dir tests/token") !=
        std::string::npos);
  CHECK(coverage.find("src/emel/text/generator/*)\n"
                      "        add_changed_shard text_generator") !=
        std::string::npos);
  CHECK(coverage.find("src/emel/embeddings/*)\n"
                      "        add_changed_shard embeddings") !=
        std::string::npos);
  CHECK(coverage.find("src/emel/logits/*|src/emel/token/*)\n"
                      "        add_changed_shard logits_and_token") !=
        std::string::npos);

  const std::string quality =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  CHECK(quality.find("tests/text/generator/*)\n"
                     "      add_test_shard text_generator") !=
        std::string::npos);
  CHECK(quality.find("tests/embeddings/*)\n"
                     "      add_test_shard embeddings") !=
        std::string::npos);
  CHECK(quality.find("tests/logits/*|tests/token/*)\n"
                     "      add_test_shard logits_and_token") !=
        std::string::npos);
  CHECK(quality.find("src/emel/text/generator/*)\n"
                     "      add_test_shard text_generator") !=
        std::string::npos);
  CHECK(quality.find("src/emel/embeddings/*)\n"
                     "      add_test_shard embeddings") !=
        std::string::npos);
  CHECK(quality.find("src/emel/logits/*|src/emel/token/*)\n"
                     "      add_test_shard logits_and_token") !=
        std::string::npos);
}

TEST_CASE("coverage script enforces thresholds on changed executable lines") {
  const std::string script =
      read_file(repo_root() / "scripts" / "test_with_coverage.sh");

  CHECK(script.find("COVERAGE_CHANGED_LINE_ONLY") != std::string::npos);
  CHECK(script.find("required_tools+=(python3)") != std::string::npos);
  CHECK(script.find("collect_changed_lines()") != std::string::npos);
  CHECK(script.find("enforce_changed_line_coverage()") != std::string::npos);
  CHECK(script.find("--json \"$coverage_json\"") != std::string::npos);
  CHECK(script.find("changed-line coverage:") != std::string::npos);
  CHECK(script.find("--fail-under-line \"$LINE_COVERAGE_MIN\"") !=
        std::string::npos);
  CHECK(script.find("--fail-under-branch \"$BRANCH_COVERAGE_MIN\"") !=
        std::string::npos);
}

TEST_CASE("coverage reports merge template instantiations by source line") {
  const std::string script =
      read_file(repo_root() / "scripts" / "test_with_coverage.sh");
  const std::size_t threshold_start =
      script.find("enforcing coverage thresholds:");
  REQUIRE(threshold_start != std::string::npos);

  const std::size_t json_report_start =
      script.find("  gcovr \\", threshold_start);
  REQUIRE(json_report_start != std::string::npos);
  const std::size_t text_report_start =
      script.find("  gcovr \\", json_report_start + 1);
  REQUIRE(text_report_start != std::string::npos);

  const std::string json_report =
      script.substr(json_report_start, text_report_start - json_report_start);
  const std::string text_report = script.substr(text_report_start);

  CHECK(json_report.find("--merge-lines") != std::string::npos);
  CHECK(json_report.find("--json \"$coverage_json\"") != std::string::npos);
  CHECK(json_report.find("suspicious_hits.warn_once_per_file") !=
        std::string::npos);

  CHECK(text_report.find("--merge-lines") != std::string::npos);
  CHECK(text_report.find("--txt-summary") != std::string::npos);
  CHECK(text_report.find("suspicious_hits.warn_once_per_file") !=
        std::string::npos);
  CHECK(text_report.find("--fail-under-line \"$LINE_COVERAGE_MIN\"") !=
        std::string::npos);
  CHECK(text_report.find("--fail-under-branch \"$BRANCH_COVERAGE_MIN\"") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates consume benchmark dependency manifest conservatively") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("BENCH_DEPENDENCY_MANIFEST_BASELINE") != std::string::npos);
  CHECK(script.find("tools/bench/dependency_manifest.txt") !=
        std::string::npos);
  CHECK(script.find("BENCH_RUNNER_BINARY") != std::string::npos);
  CHECK(script.find("bench_dependency_manifest_apply_changed_files()") !=
        std::string::npos);
  CHECK(script.find("add_all_benchmark_suites_from_manifest()") !=
        std::string::npos);
  CHECK(script.find("bench_dependency_manifest_requires_full_gate()") !=
        std::string::npos);
  CHECK(script.find("--write-dependency-manifest") != std::string::npos);
  CHECK(script.find("--check-dependency-manifest") != std::string::npos);
  CHECK(script.find("dependency manifest requires full benchmark gate") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates consume parity dependency manifest for runner selection") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("PARITY_DEPENDENCY_MANIFEST_BASELINE") !=
        std::string::npos);
  CHECK(script.find("tools/paritychecker/dependency_manifest.txt") !=
        std::string::npos);
  CHECK(script.find("parity_dependency_manifest_apply_changed_files()") !=
        std::string::npos);
  CHECK(script.find("parity_toolchain_file_requires_full_gate()") !=
        std::string::npos);
  CHECK(script.find("select_full_parity_gate \"paritychecker toolchain change "
                    "path=$file\"") != std::string::npos);
  CHECK(script.find("add_parity_runner \"$runner\" \"manifest path=$path\"") !=
        std::string::npos);
  CHECK(script.find("select_full_parity_gate \"unmatched parity-relevant "
                    "change path=$file\"") != std::string::npos);
  CHECK(script.find("scripts/paritychecker.sh\" \"${runner_args[@]}\"") !=
        std::string::npos);
}

TEST_CASE("quality gates check parity manifest freshness before deciding skip "
          "branch") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t run_start = script.find("run_parity_gate()");
  REQUIRE(run_start != std::string::npos);

  const std::size_t case_branch =
      script.find("case \"$QUALITY_GATES_PARITY\" in", run_start);
  REQUIRE(case_branch != std::string::npos);

  const std::string pre_case =
      script.substr(run_start, case_branch - run_start);
  CHECK(pre_case.find("[[ \"$QUALITY_GATES_PARITY\" != \"always\" ]]") !=
        std::string::npos);
  CHECK(pre_case.find("parity_dependency_manifest_requires_full_gate") !=
        std::string::npos);
  CHECK(pre_case.find(
            "select_full_parity_gate \"dependency manifest freshness gap\"") !=
        std::string::npos);
  CHECK(pre_case.find("parity_dependency_manifest_check_needed") ==
        std::string::npos);
}

TEST_CASE("paritychecker script supports selected maintained runners") {
  const std::string script =
      read_file(repo_root() / "scripts" / "paritychecker.sh");

  CHECK(script.find("--runner=<name>|--mode=<name>") != std::string::npos);
  CHECK(script.find("selected_runners=()") != std::string::npos);
  CHECK(script.find("gbnf)\n      runner=\"gbnf_parser\"") !=
        std::string::npos);
  CHECK(script.find("paritychecker: runner=$runner") != std::string::npos);
  CHECK(script.find("--test-case=\"*tokens across tiny models*\"") !=
        std::string::npos);
  CHECK(script.find("--test-case=\"*gbnf parser outputs*\"") !=
        std::string::npos);
  CHECK(script.find("--test-case=\"*kernel outputs*\"") != std::string::npos);
  CHECK(script.find("--test-case=\"*jinja parser and formatter outputs*\"") !=
        std::string::npos);
  CHECK(script.find(
            "--test-case=\"paritychecker matches current maintained generation "
            "publication against live reference\"") != std::string::npos);
}

TEST_CASE("quality gates preserve failing lane status in parallel children") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("run_step()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("run_step_allow_fail()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("local status=0") != std::string::npos);
  CHECK(helper.find("if \"$@\"; then") != std::string::npos);
  CHECK(helper.find("status=$?") != std::string::npos);
  CHECK(helper.find("return \"$status\"") != std::string::npos);
}

TEST_CASE(
    "quality gates can run independent heavy lanes in ordered parallel group") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("QUALITY_GATES_PARALLEL") != std::string::npos);
  CHECK(script.find("parallel_enabled()") != std::string::npos);
  CHECK(script.find("EMEL_BENCH_STRICT_REGRESSION") != std::string::npos);
  CHECK(script.find("start_parallel_step bench_snapshot run_benchmark_gates") !=
        std::string::npos);
  CHECK(
      script.find("start_parallel_step test_with_coverage run_coverage_gate") !=
      std::string::npos);
  CHECK(script.find("start_parallel_step paritychecker run_parity_gate") !=
        std::string::npos);
  CHECK(script.find("start_parallel_step fuzz_smoke run_fuzz_gate") !=
        std::string::npos);
  CHECK(script.find("quality_gates: log begin name=$name") !=
        std::string::npos);
  CHECK(script.find("quality_gates: log end name=$name status=$status") !=
        std::string::npos);
  CHECK(script.find("EMEL_QUALITY_GATES_PARALLEL_CHILD=1") !=
        std::string::npos);
  CHECK(script.find("set +e\n    \"${lane_cmd[@]}\" >>\"$log_file\" 2>&1") !=
        std::string::npos);
  CHECK(script.find("printf '%s\\n' \"$status\" >\"$status_file\"") !=
        std::string::npos);
}

TEST_CASE("quality gate defaults are nounset-safe and use canonical evaluator path") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find(
            "QUALITY_GATES_FUZZ=\"${EMEL_QUALITY_GATES_FUZZ:-auto}\"") !=
        std::string::npos);
  CHECK(script.find("QUALITY_GATES_DETERMINISM=\"${EMEL_QUALITY_GATES_"
                    "DETERMINISM:-auto}\"") != std::string::npos);
  CHECK(script.find("QUALITY_GATES_PARALLEL=\"${EMEL_QUALITY_GATES_"
                    "PARALLEL:-auto}\"") != std::string::npos);
  CHECK(script.find("build/zig/needle_eval/emel_needle_eval") !=
        std::string::npos);
}

TEST_CASE("quality gates preserve fractional and disabled timeout budgets") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("timeout_seconds()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("lane_timeout_for()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string timeout_body =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(timeout_body.find("([.][0-9]+)?") != std::string::npos);
  CHECK(timeout_body.find("suffix == \"m\"") != std::string::npos);
  CHECK(timeout_body.find("printf \"%.3f\\n\"") != std::string::npos);

  const std::size_t lane_start = script.find("lane_timeout_for()");
  REQUIRE(lane_start != std::string::npos);

  const std::size_t lane_end =
      script.find("run_domain_boundary_gate()", lane_start);
  REQUIRE(lane_end != std::string::npos);

  const std::string lane_body =
      script.substr(lane_start, lane_end - lane_start);
  CHECK(lane_body.find("global_seconds <= 0") != std::string::npos);
  CHECK(lane_body.find("print \"0\"") != std::string::npos);
  CHECK(lane_body.find("budget = global_seconds - elapsed - flush_margin") !=
        std::string::npos);
  CHECK(script.find("lane_timeout_enabled()") != std::string::npos);
  CHECK(script.find("lane_timeout_enabled \"$budget\"") != std::string::npos);
  CHECK(script.find("exit !(budget > 0)") != std::string::npos);
}

TEST_CASE("quality gate script changes keep mandatory lanes conservative") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  const std::size_t infer_start = script.find("infer_quality_gate_scope()");
  REQUIRE(infer_start != std::string::npos);

  const std::size_t infer_end =
      script.find("if [[ \"$QUALITY_GATES_PARITY\"", infer_start);
  REQUIRE(infer_end != std::string::npos);

  const std::string infer_body =
      script.substr(infer_start, infer_end - infer_start);
  CHECK(infer_body.find("scripts/quality_gates.sh)") != std::string::npos);
  CHECK(infer_body.find("coverage_all_required=true") != std::string::npos);
  CHECK(infer_body.find("select_full_parity_gate \"quality gate script changed "
                        "path=$file\"") != std::string::npos);
  CHECK(infer_body.find("quality_gates: select benchmark runner=all "
                        "reason=quality gate script changed path=$file") !=
        std::string::npos);
  CHECK(infer_body.find("bench_all_suites=true") != std::string::npos);
  CHECK(infer_body.find("add_all_benchmark_suites_from_manifest") !=
        std::string::npos);
  CHECK(script.find("\"$QUALITY_GATES_SCOPE\" == \"full\" || "
                    "\"$coverage_all_required\" == \"true\"") !=
        std::string::npos);
}

TEST_CASE("bench script keeps suite-filtered builds out of canonical "
          "bench-tools cache") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("bench_suite_build_dir()") != std::string::npos);
  CHECK(script.find("build/bench_tools_ninja_${safe_suite}") !=
        std::string::npos);
  CHECK(script.find("BENCH_COMPARE_BUILD_DIR:-$(bench_suite_build_dir") !=
        std::string::npos);
  CHECK(script.find("BENCH_BUILD_DIR:-$(bench_suite_build_dir") !=
        std::string::npos);
}

TEST_CASE("bench script exposes unfiltered bench tool validation command") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("--test-tools") != std::string::npos);
  CHECK(script.find("--test-tools cannot be combined") != std::string::npos);
  CHECK(script.find(
            "BENCH_TOOLS_TEST_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja") !=
        std::string::npos);
  CHECK(script.find("bench_runner_tests quality_gates_tests") !=
        std::string::npos);
  const std::string ctest_contract = "ctest --test-dir \"$build_dir\" -R "
                                     "'quality_gates_tests|bench_runner_tests'";
  CHECK(script.find(ctest_contract) != std::string::npos);
}

TEST_CASE("quality gates wire the maintained Needle evaluator with release-safe resources") {
  const std::string cmake = read_file(repo_root() / "CMakeLists.txt");
  CHECK(cmake.find("EMEL_ENABLE_NEEDLE_EVAL") != std::string::npos);
  CHECK(cmake.find("add_subdirectory(tools/needle_eval needle_eval)") !=
        std::string::npos);

  const std::string evaluator =
      read_file(repo_root() / "tools" / "needle_eval" / "main.cpp");
  CHECK(evaluator.find("vocab->bos_id") != std::string::npos);
  CHECK(evaluator.find("vocab->eos_id") != std::string::npos);
  CHECK(evaluator.find("min_domain_accuracy = 0.840") != std::string::npos);
  CHECK(evaluator.find("min_effort_accuracy = 0.760") != std::string::npos);
  CHECK(evaluator.find("max_no_parse = 1u") != std::string::npos);
  CHECK(evaluator.find("comparison_precision=3dp") != std::string::npos);
  CHECK(evaluator.find("domain_acc=%.4f") != std::string::npos);
  CHECK(evaluator.find("effort_acc=%.4f") != std::string::npos);
  CHECK(evaluator.find("accuracy thresholds unmet") != std::string::npos);

  const std::string quality =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  CHECK(quality.find("run_needle_eval_gate()") != std::string::npos);
  CHECK(quality.find("needle_eval required resources missing:") !=
        std::string::npos);
  CHECK(quality.find("QUALITY_GATES_SCOPE\" == \"full\"") !=
        std::string::npos);
  CHECK(quality.find("--min-domain-accuracy") != std::string::npos);
  CHECK(quality.find("--min-effort-accuracy") != std::string::npos);
  CHECK(quality.find("--max-no-parse") != std::string::npos);
  CHECK(quality.find(
            "QUALITY_GATES_NEEDLE_MAX_NO_PARSE=\"${EMEL_QUALITY_GATES_NEEDLE_MAX_NO_PARSE:-1}\"") !=
        std::string::npos);
  CHECK(quality.find("--activation-route f32") != std::string::npos);

  CHECK(evaluator.find("emel/io/mmap/sm.hpp") != std::string::npos);
  CHECK(evaluator.find("emel::io::mmap::event::map_tensor_request") !=
        std::string::npos);
  CHECK(evaluator.find("emel::io::mmap::event::release_mapping") !=
        std::string::npos);
  CHECK(evaluator.find("read_file_bytes") == std::string::npos);
  CHECK(evaluator.find("k_max_model_bytes") != std::string::npos);
  CHECK(evaluator.find("k_max_tsv_bytes") != std::string::npos);
  CHECK(evaluator.find("k_max_tsv_rows") != std::string::npos);
  CHECK(evaluator.find("k_max_tsv_line_bytes") != std::string::npos);
  CHECK(evaluator.find("k_max_reference_ids_per_row") != std::string::npos);
  CHECK(evaluator.find("std::strtoul") != std::string::npos);
  CHECK(evaluator.find("std::bad_alloc") != std::string::npos);
}

TEST_CASE("Needle evaluator defaults to heldout f32 and retains explicit A8") {
  const std::string evaluator =
      read_file(repo_root() / "tools" / "needle_eval" / "main.cpp");
  const std::string graph_events = read_file(
      repo_root() / "src" / "emel" / "model" / "needle" / "graph" /
      "events.hpp");

  CHECK(evaluator.find("enum class activation_route { a8, f32 }") !=
        std::string::npos);
  CHECK(evaluator.find("activation_route route = activation_route::f32") !=
        std::string::npos);
  CHECK(evaluator.find("--activation-route") != std::string::npos);
  CHECK(evaluator.find("value == \"a8\"") != std::string::npos);
  CHECK(evaluator.find("value == \"f32\"") != std::string::npos);
  CHECK(evaluator.find(
            ".activation_quant = options.route == activation_route::a8") !=
        std::string::npos);
  CHECK(evaluator.find("activation_route_name(options.route)") !=
        std::string::npos);
  CHECK(evaluator.find("activation_route=%s") != std::string::npos);
  CHECK(graph_events.find("bool activation_quant = false") !=
        std::string::npos);
  CHECK(read_file(repo_root() / "tests" / "model" / "needle" /
                  "graph_parity_tests.cpp")
            .find("graph::event::init{true}") != std::string::npos);
}

TEST_CASE("Needle evaluator compares published accuracy at three decimals") {
  const std::string evaluator =
      read_file(repo_root() / "tools" / "needle_eval" / "main.cpp");

  CHECK(evaluator.find(
            "static_assert(!accuracy_meets_threshold(0.8394, 0.840))") !=
        std::string::npos);
  CHECK(evaluator.find(
            "static_assert(accuracy_meets_threshold(0.8395, 0.840))") !=
        std::string::npos);
  CHECK(evaluator.find(
            "static_assert(!accuracy_meets_threshold(0.7594, 0.760))") !=
        std::string::npos);
  CHECK(evaluator.find(
            "static_assert(accuracy_meets_threshold(0.7595, 0.760))") !=
        std::string::npos);
  CHECK(evaluator.find("static_assert(eval_options{}.max_no_parse == 1u)") !=
        std::string::npos);
}

#if !defined(_WIN32) && defined(BENCH_NEEDLE_EVAL_BINARY)
TEST_CASE(
    "Needle evaluator rejects invalid activation routes before evaluation") {
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() /
      ("needle_eval_activation_route_" + std::to_string(std::rand()) +
       ".txt");
  const command_result result = run_command(
      shell_quote(BENCH_NEEDLE_EVAL_BINARY) +
          " unused.cact unused.tsv --activation-route invalid",
      output);

  CHECK(result.status != 0);
  CHECK(result.output.find("invalid CLI option or value") !=
        std::string::npos);
}

TEST_CASE("Needle evaluator accepts maintained activation routes before model "
          "evaluation") {
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() /
      ("needle_eval_activation_route_valid_" + std::to_string(std::rand()) +
       ".txt");

  for (const char *route : {"a8", "f32"}) {
    const command_result result = run_command(
        shell_quote(BENCH_NEEDLE_EVAL_BINARY) +
            " unused.cact unused.tsv --activation-route " + route,
        output);
    CHECK(result.status != 0);
    CHECK(result.output.find("open prompts tsv") != std::string::npos);
    CHECK(result.output.find("invalid CLI option or value") ==
          std::string::npos);
  }
}

TEST_CASE("Needle evaluator rejects malformed IDs before model evaluation") {
  static int fixture_counter = 0;
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("needle_eval_contract_" + std::to_string(++fixture_counter) + "_" +
       std::to_string(std::rand()));
  std::filesystem::create_directories(dir);
  const std::filesystem::path prompts = dir / "prompts.tsv";
  const std::filesystem::path output = dir / "output.txt";
  write_file(prompts, "routing\tlow\t1 junk\t6869\n");

  const command_result result = run_command(
      shell_quote(BENCH_NEEDLE_EVAL_BINARY) + " " +
          shell_quote((repo_root() / "tests" / "models" /
                       "route-w4-qat.cact")
                          .string()) +
          " " + shell_quote(prompts.string()) + " 0 1",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("invalid reference IDs") != std::string::npos);
  CHECK(result.output.find("native tokenizer ids differ") ==
        std::string::npos);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE("Needle evaluator rejects oversized resources before evaluation") {
  static int fixture_counter = 0;
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("needle_eval_bounds_" + std::to_string(++fixture_counter) + "_" +
       std::to_string(std::rand()));
  std::filesystem::create_directories(dir);
  const std::filesystem::path prompts = dir / "prompts.tsv";
  const std::filesystem::path output = dir / "output.txt";
  write_file(prompts, "routing\tlow\t1\t" + std::string(65538u, '0') + "\n");

  const command_result result = run_command(
      shell_quote(BENCH_NEEDLE_EVAL_BINARY) + " " +
          shell_quote((repo_root() / "tests" / "models" /
                       "route-w4-qat.cact")
                          .string()) +
          " " + shell_quote(prompts.string()) + " 0 1",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("prompts tsv line too large") !=
        std::string::npos);
  CHECK(result.output.find("row i=") == std::string::npos);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
#endif


TEST_CASE("bench script routes Moshi LM suite through the wrapper") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");
  const std::size_t route_start =
      script.find("[[ \"$SUITE_FILTER\" == \"speech_lm_moshi\" ]]");
  REQUIRE(route_start != std::string::npos);
  const std::size_t route_end = script.find("prepare_toolchain()", route_start);
  REQUIRE(route_end != std::string::npos);
  const std::string route = script.substr(route_start, route_end - route_start);

  CHECK(route.find("speech_lm_moshi has no reference lane") !=
        std::string::npos);
  CHECK(route.find("bash \"$ROOT_DIR/scripts/bench_moshi_lm_compare.sh\"") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates route PersonaPlex changes through end-to-end compare") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("add_bench_suite speech_dialogue_moshi") !=
        std::string::npos);
  CHECK(script.find("speech_dialogue_moshi)") != std::string::npos);
  CHECK(script.find("scripts/bench_personaplex_compare.sh") !=
        std::string::npos);
  CHECK(script.find("QUALITY_GATES_PERSONAPLEX_FRAMES") != std::string::npos);
  CHECK(script.find("QUALITY_GATES_PERSONAPLEX_AUDIO") != std::string::npos);
  CHECK(script.find("--audio \"$QUALITY_GATES_PERSONAPLEX_AUDIO\"") !=
        std::string::npos);
  CHECK(script.find("tools/bench/moshi_gguf_convert.py|") != std::string::npos);
  CHECK(script.find("scripts/setup_moshi_cpp_reference.sh|") !=
        std::string::npos);
}

TEST_CASE("benchmark defaults stay bounded for routine quality gates") {
  const std::string quality_gates =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::string bench_runner =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");

  CHECK(quality_gates.find("QUALITY_GATES_BENCH_ITERS=\"${EMEL_QUALITY_GATES_"
                           "BENCH_ITERS:-100}\"") != std::string::npos);
  CHECK(
      quality_gates.find(
          "QUALITY_GATES_BENCH_RUNS=\"${EMEL_QUALITY_GATES_BENCH_RUNS:-3}\"") !=
      std::string::npos);
  CHECK(quality_gates.find("QUALITY_GATES_BENCH_WARMUP_ITERS=\"${EMEL_QUALITY_"
                           "GATES_BENCH_WARMUP_ITERS:-10}\"") !=
        std::string::npos);

  CHECK(bench_runner.find(
            "constexpr std::uint64_t k_default_iterations = 100;") !=
        std::string::npos);
  CHECK(bench_runner.find("constexpr std::size_t k_default_runs = 3;") !=
        std::string::npos);
  CHECK(bench_runner.find(
            "constexpr std::uint64_t k_default_warmup_iterations = 10;") !=
        std::string::npos);
}

TEST_CASE("bench script bounds default generation workload") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");
  const std::string generation_bench =
      read_file(repo_root() / "tools" / "bench" / "generation_bench.cpp");
  const std::string diarization_bench = read_file(
      repo_root() / "tools" / "bench" / "diarization" / "sortformer_bench.cpp");

  CHECK(script.find("DEFAULT_GENERATION_WORKLOAD_ID=") != std::string::npos);
  CHECK(script.find("lfm2_single_user_hello_max_tokens_1_v1") !=
        std::string::npos);
  CHECK(
      script.find("EMEL_GENERATION_WORKLOAD_ID=\"$generation_workload_id\"") !=
      std::string::npos);
  CHECK(script.find("DEFAULT_DIARIZATION_ITERS=\"${EMEL_BENCH_DEFAULT_"
                    "DIARIZATION_ITERS:-1}\"") != std::string::npos);
  CHECK(script.find("DEFAULT_DIARIZATION_RUNS=\"${EMEL_BENCH_DEFAULT_"
                    "DIARIZATION_RUNS:-3}\"") != std::string::npos);
  CHECK(script.find("EMEL_BENCH_DIARIZATION_ITERS=\"$diarization_iters\"") !=
        std::string::npos);
  CHECK(script.find("EMEL_BENCH_DIARIZATION_RUNS=\"$diarization_runs\"") !=
        std::string::npos);
  CHECK(script.find("TOLERANCE=\"${BENCH_TOLERANCE:-0.30}\"") !=
        std::string::npos);
  CHECK(script.find("ABS_TOLERANCE_NS=\"${BENCH_ABS_TOLERANCE_NS:-5000}\"") !=
        std::string::npos);
  // The regression-comparison logic lives in the extracted compare gate.
  const std::string compare_gate =
      read_file(repo_root() / "scripts" / "bench_compare_gate.awk");
  CHECK(compare_gate.find(
            "curr[name] > relative_limit && curr[name] > absolute_limit") !=
        std::string::npos);
  CHECK(generation_bench.find("filter.empty() || filter == \"all\"") !=
        std::string::npos);
  CHECK(diarization_bench.find("EMEL_BENCH_DIARIZATION_ITERS") !=
        std::string::npos);
  CHECK(diarization_bench.find("EMEL_BENCH_DIARIZATION_RUNS") !=
        std::string::npos);
}

TEST_CASE("bench combined gate snapshots EMEL before paired comparison") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");
  const std::size_t combined_start = script.find("if $COMBINED; then");
  REQUIRE(combined_start != std::string::npos);
  const std::size_t combined_end = script.find("if $SNAPSHOT; then", combined_start);
  REQUIRE(combined_end != std::string::npos);
  const std::string combined =
      script.substr(combined_start, combined_end - combined_start);

  CHECK(combined.find("run_bench_runner \"$build_dir\" --mode=emel > "
                      "\"$snapshot_output\"") != std::string::npos);
  CHECK(combined.find("run_bench_runner \"$build_dir\" --mode=compare > "
                      "\"$compare_output\"") != std::string::npos);
  CHECK(combined.find("filter_snapshot_regression_rows \"$snapshot_output\" "
                      "\"$current_snapshot\"") != std::string::npos);
  CHECK(combined.find("snapshot_gate_has_only_live_needle_diagnostics "
                      "\"$snapshot_output\"") != std::string::npos);
  CHECK(combined.find("snapshot_measurement_rows=\"$(awk '") !=
        std::string::npos);
  CHECK(combined.find("elif [[ -z \"$SUITE_FILTER\" || "
                      "\"$snapshot_measurement_rows\" == \"0\" ]]") !=
        std::string::npos);
}

TEST_CASE(
    "bench script merges scoped snapshot updates into the full baseline") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("update_snapshot_baseline()") != std::string::npos);
  CHECK(script.find("if [[ -z \"$SUITE_FILTER\" ]]") != std::string::npos);
  CHECK(script.find("-v ref=\"$ref_value\" -v toolchain=\"$bench_cxx\"") !=
        std::string::npos);
  CHECK(script.find("print \"# ref=\" ref") != std::string::npos);
  CHECK(script.find("print \"# toolchain=\" toolchain") != std::string::npos);
  CHECK(script.find("curr[name] = $0") != std::string::npos);
  CHECK(script.find("order[++order_count] = name") != std::string::npos);
  CHECK(script.find("updated $baseline (merged suite $SUITE_FILTER)") !=
        std::string::npos);
  CHECK(script.find(
            "update_snapshot_baseline \"$BASELINE\" \"$current_snapshot\"") !=
        std::string::npos);
  CHECK(script.find("update_snapshot_baseline \"$BASELINE\" \"$CURRENT\"") !=
        std::string::npos);
  CHECK(script.find("tokens_per_second = \"\"") != std::string::npos);
  CHECK(script.find("tokens_per_second=%s") != std::string::npos);
}

TEST_CASE("bench script rejects scoped compare baseline updates") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("if $COMPARE_UPDATE && [[ -n \"$SUITE_FILTER\" ]]") !=
        std::string::npos);
  CHECK(script.find("--compare-update cannot be combined with --suite or "
                    "--generation-only") != std::string::npos);
}

TEST_CASE("quality gates map benchmark manifest records to scoped or full "
          "benchmark gates") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start =
      script.find("bench_dependency_manifest_apply_changed_files()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("collect_changed_files()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("runner=\"\"") != std::string::npos);
  CHECK(helper.find("path=\"\"") != std::string::npos);
  CHECK(helper.find("runner=*)") != std::string::npos);
  CHECK(helper.find("path=*)") != std::string::npos);
  CHECK(helper.find("bench_all_suites=true") != std::string::npos);
  CHECK(helper.find("add_all_benchmark_suites_from_manifest") !=
        std::string::npos);
  CHECK(helper.find("add_bench_suite \"$runner\"") != std::string::npos);
  CHECK(helper.find("tools/bench/*|tools/bench/**/*") != std::string::npos);
}

TEST_CASE(
    "quality gates expand full benchmark scope into all manifest suites") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start =
      script.find("add_all_benchmark_suites_from_manifest()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find(
      "bench_dependency_manifest_record_matches_file()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("add_benchmark_suite_from_manifest \"$priority_runner\"") !=
        std::string::npos);
  CHECK(helper.find("gbnf_rule_parser") != std::string::npos);
  CHECK(helper.find("kernel_aarch64") != std::string::npos);
  const std::size_t parallel_matmul_priority = helper.find("parallel_matmul");
  const std::size_t manifest_expansion = helper.find("while IFS= read -r line");
  REQUIRE(parallel_matmul_priority != std::string::npos);
  REQUIRE(manifest_expansion != std::string::npos);
  CHECK(parallel_matmul_priority < manifest_expansion);
  CHECK(helper.find("runner=\"\"") != std::string::npos);
  CHECK(helper.find("runner=*)") != std::string::npos);
  CHECK(helper.find("\"$runner\" == \"all\"") != std::string::npos);
  CHECK(helper.find("\"$runner\" == \"weight_streaming\"") !=
        std::string::npos);
  CHECK(helper.find("EMEL_BENCH_WEIGHT_STREAMING") != std::string::npos);
  CHECK(helper.find("bench_suite_supported_for_host \"$runner\"") !=
        std::string::npos);
  CHECK(helper.find("add_bench_suite \"$runner\"") != std::string::npos);

  const std::size_t infer_start = script.find("infer_quality_gate_scope()");
  REQUIRE(infer_start != std::string::npos);
  const std::size_t infer_end =
      script.find("collect_changed_files", infer_start);
  REQUIRE(infer_end != std::string::npos);
  const std::string full_scope =
      script.substr(infer_start, infer_end - infer_start);
  CHECK(full_scope.find("bench_all_suites=true") != std::string::npos);
  CHECK(full_scope.find("add_all_benchmark_suites_from_manifest") !=
        std::string::npos);
}

TEST_CASE("quality gates skip host-incompatible benchmark suites during full "
          "expansion") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start =
      script.find("bench_suite_supported_for_host()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("add_all_benchmark_suites_from_manifest()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("kernel_x86_64)") != std::string::npos);
  CHECK(helper.find("\"x86_64\"") != std::string::npos);
  CHECK(helper.find("\"amd64\"") != std::string::npos);
  CHECK(helper.find("kernel_aarch64)") != std::string::npos);
  CHECK(helper.find("\"aarch64\"") != std::string::npos);
  CHECK(helper.find("\"arm64\"") != std::string::npos);
  CHECK(helper.find("sm_any|sm_scheduler)") != std::string::npos);
  CHECK(helper.find("EMEL_BENCH_INTERNAL") != std::string::npos);
}

TEST_CASE("quality gates enable internal env for selected internal benchmark "
          "suites") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t run_start = script.find("run_benchmark_gates()");
  REQUIRE(run_start != std::string::npos);

  const std::size_t run_end = script.find("run_coverage_gate()", run_start);
  REQUIRE(run_end != std::string::npos);

  const std::string run_body = script.substr(run_start, run_end - run_start);
  CHECK(run_body.find("sm_any|sm_scheduler)") != std::string::npos);
  CHECK(run_body.find("bench_extra_env+=(EMEL_BENCH_INTERNAL=1)") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates check benchmark manifest before deciding benchmark branch") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t run_start = script.find("run_benchmark_gates()");
  REQUIRE(run_start != std::string::npos);

  const std::size_t full_branch =
      script.find("if $bench_full; then", run_start);
  REQUIRE(full_branch != std::string::npos);

  const std::string pre_full =
      script.substr(run_start, full_branch - run_start);
  CHECK(pre_full.find("bench_dependency_manifest_check_needed") !=
        std::string::npos);
  CHECK(pre_full.find("bench_dependency_manifest_requires_full_gate") !=
        std::string::npos);
  CHECK(pre_full.find("bench_full=true") != std::string::npos);
  CHECK(pre_full.find("if ! $bench_full && $bench_all_suites") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates dispatch opt-in suites during full benchmark fallback") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t selector_start =
      script.find("benchmark_suite_selected_for_full_dispatch()");
  REQUIRE(selector_start != std::string::npos);
  const std::size_t selector_end =
      script.find("run_full_benchmark_opt_in_suites()", selector_start);
  REQUIRE(selector_end != std::string::npos);
  const std::string selector =
      script.substr(selector_start, selector_end - selector_start);
  CHECK(selector.find("if $bench_full; then") != std::string::npos);
  CHECK(selector.find("return 0") != std::string::npos);

  const std::size_t helper_start =
      script.find("run_full_benchmark_opt_in_suites()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("run_fuzz_gate()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("for suite in speech_lm_moshi") != std::string::npos);
  CHECK(helper.find("benchmark_suite_selected_for_full_dispatch \"$suite\"") !=
        std::string::npos);
  CHECK(helper.find("\"bench_snapshot_${suite}\"") != std::string::npos);
  CHECK(helper.find("--suite=\"$suite\"") != std::string::npos);

  const std::size_t run_start = script.find("run_benchmark_gates()");
  REQUIRE(run_start != std::string::npos);
  const std::size_t run_end = script.find("run_coverage_gate()", run_start);
  REQUIRE(run_end != std::string::npos);
  const std::string run_body = script.substr(run_start, run_end - run_start);
  CHECK(run_body.find("run_full_benchmark_opt_in_suites") != std::string::npos);
}

TEST_CASE("quality gates expand broad benchmark scope without monolithic "
          "changed gate") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("select_full_benchmark_gate()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("add_parity_runner()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper =
      script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("\"$QUALITY_GATES_SCOPE\" == \"full\"") !=
        std::string::npos);
  CHECK(helper.find("bench_all_suites=true") != std::string::npos);
  CHECK(helper.find("add_all_benchmark_suites_from_manifest") !=
        std::string::npos);
}

TEST_CASE(
    "quality gates bound scoped generation benchmark workload explicitly") {
  const std::string script =
      read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("QUALITY_GATES_DEFAULT_GENERATION_WORKLOAD_ID") !=
        std::string::npos);
  CHECK(script.find("lfm2_single_user_hello_max_tokens_1_v1") !=
        std::string::npos);
  CHECK(script.find("EMEL_BENCH_GENERATION_ITERS") != std::string::npos);
  CHECK(script.find("EMEL_BENCH_GENERATION_RUNS") != std::string::npos);
  CHECK(
      script.find("EMEL_GENERATION_WORKLOAD_ID=\"$generation_workload_id\"") !=
      std::string::npos);
  CHECK(script.find("\"$generation_workload_id\" != \"all\"") ==
        std::string::npos);
}

TEST_CASE("bench runner generation tests use a bounded workload filter") {
  const std::string tests =
      read_file(repo_root() / "tools" / "bench" / "bench_runner_tests.cpp");

  CHECK(tests.find("k_bounded_generation_workload_id") != std::string::npos);
  CHECK(tests.find("lfm2_single_user_hello_max_tokens_1_v1") !=
        std::string::npos);
  CHECK(tests.find("EMEL_GENERATION_WORKLOAD_ID=") != std::string::npos);
}

#if !defined(_WIN32)
// A baseline that carries both paired and host-renamed arch rows, mirroring the
// committed snapshots/bench/benchmarks.txt shape. The runner only emits rows
// for the host arch (case_supported_on_host), so a full unscoped compare on a
// single-arch host must not demand foreign-arch rows or reject the host-renamed
// counterpart of a foreign baseline row.
const std::string k_dual_arch_baseline =
    "# ref=test\n"
    "batch/planner_simple ns_per_op=650.000 iter=100 runs=3\n"
    "flash_attention/aarch64/op_flash_attn_ext_decode_like ns_per_op=17892.000 "
    "iter=100 runs=3\n"
    "kernel/aarch64/op_add ns_per_op=127.000 iter=100 runs=3\n"
    "kernel/x86_64/op_add ns_per_op=122.000 iter=100 runs=3\n"
    "kernel/x86_64/op_flash_attn_ext_decode_like ns_per_op=172.000 iter=100 "
    "runs=3\n";

TEST_CASE(
    "compare gate exempts foreign-arch baseline rows the host runner skips") {
  // Current snapshot as an arm64 runner would emit it: no x86_64 rows.
  const std::string arm64_current =
      "# bench_host_arch: aarch64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "flash_attention/aarch64/op_flash_attn_ext_decode_like "
      "ns_per_op=17895.000\n"
      "kernel/aarch64/op_add ns_per_op=128.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, arm64_current, "aarch64") == 0);

  // Symmetric case: an x86_64 host omits the aarch64 rows and still passes.
  const std::string x86_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "flash_attention/x86_64/op_flash_attn_ext_decode_like ns_per_op=170.000\n"
      "kernel/x86_64/op_flash_attn_ext_decode_like ns_per_op=171.000\n"
      "kernel/x86_64/op_add ns_per_op=123.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, x86_current, "x86_64") == 0);
}

TEST_CASE(
    "compare gate still fails when a host-native baseline row is missing") {
  // The gate must not be weakened: a missing row for the host's own arch, or a
  // missing arch-agnostic row, is a genuine failure.
  const std::string arm64_missing_native =
      "# bench_host_arch: aarch64\n"
      "batch/planner_simple ns_per_op=655.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, arm64_missing_native,
                         "aarch64") != 0);

  const std::string arm64_missing_shared =
      "# bench_host_arch: aarch64\n"
      "kernel/aarch64/op_add ns_per_op=128.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, arm64_missing_shared,
                         "aarch64") != 0);
}

TEST_CASE(
    "compare gate still fails when a host-renamed row has no paired baseline") {
  const std::string x86_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "flash_attention/x86_64/untracked_case ns_per_op=170.000\n"
      "kernel/x86_64/op_add ns_per_op=123.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, x86_current, "x86_64") != 0);
}

TEST_CASE(
    "compare gate does not borrow a foreign baseline for host-renamed rows") {
  const std::string foreign_only_baseline =
      "# ref=test\n"
      "batch/planner_simple ns_per_op=650.000 iter=100 runs=3\n"
      "flash_attention/aarch64/op_flash_attn_ext_decode_like "
      "ns_per_op=17892.000 iter=100 runs=3\n";
  const std::string x86_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "flash_attention/x86_64/op_flash_attn_ext_decode_like "
      "ns_per_op=170.000\n";
  CHECK(run_compare_gate(foreign_only_baseline, x86_current, "x86_64") != 0);
}

TEST_CASE("compare gate requires explicit aliases for renamed benchmark rows") {
  const std::string unrelated_rename_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "some_suite/x86_64/op_add ns_per_op=123.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, unrelated_rename_current,
                         "x86_64") != 0);
}

TEST_CASE("compare gate treats aliased baseline rows as covered") {
  const std::string missing_kernel_flash_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "flash_attention/x86_64/op_flash_attn_ext_decode_like ns_per_op=170.000\n"
      "kernel/x86_64/op_add ns_per_op=123.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, missing_kernel_flash_current,
                         "x86_64") == 0);
}

TEST_CASE(
    "compare gate requires host counterpart before skipping foreign rows") {
  const std::string missing_flash_suite_current =
      "# bench_host_arch: x86_64\n"
      "batch/planner_simple ns_per_op=655.000\n"
      "kernel/x86_64/op_flash_attn_ext_decode_like ns_per_op=171.000\n"
      "kernel/x86_64/op_add ns_per_op=123.000\n";
  CHECK(run_compare_gate(k_dual_arch_baseline, missing_flash_suite_current,
                         "x86_64") != 0);
}
#endif

TEST_CASE("bench runner emits a host-arch marker the compare gate consumes") {
  const std::string bench_runner =
      read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");
  CHECK(bench_runner.find("constexpr std::string_view k_host_arch =") !=
        std::string::npos);
  CHECK(bench_runner.find("void print_bench_host_arch_marker()") !=
        std::string::npos);
  const std::size_t snapshot_start = bench_runner.find("void print_snapshot");
  const std::size_t compare_start = bench_runner.find("void print_compare");
  REQUIRE(snapshot_start != std::string::npos);
  REQUIRE(compare_start != std::string::npos);
  CHECK(bench_runner.find("print_bench_host_arch_marker();", snapshot_start) !=
        std::string::npos);
  CHECK(bench_runner.find("print_bench_host_arch_marker();", compare_start) !=
        std::string::npos);

  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");
  CHECK(script.find("resolve_bench_host_arch") != std::string::npos);
  CHECK(script.find("bench_compare_gate.awk") != std::string::npos);
  // The gate must be sourced from the shared file, not re-inlined, so the
  // host-arch exemption logic has a single home.
  CHECK(script.find("-v host_arch=\"$host_arch\"") != std::string::npos);
}

#if !defined(_WIN32)
TEST_CASE("memory envelope computes half of effective total memory") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_cap_half_test.txt";
  const auto script = repo_root() / "scripts" / "build_jobs.sh";
  const command_result result = run_command(
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=max "
      "EMEL_MEMORY_TEST_CURRENT_MAX=max EMEL_MEMORY_TEST_CURRENT_SWAP=max "
      "EMEL_MEMORY_TEST_CORES=64 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status == 0);
  CHECK(result.output.find("effective_total_bytes=107374182400") !=
        std::string::npos);
  CHECK(result.output.find("cap_bytes=53687091200") != std::string::npos);
}

TEST_CASE("memory envelope honors a finite parent cgroup limit") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_cap_cgroup_test.txt";
  const auto script = repo_root() / "scripts" / "build_jobs.sh";
  const command_result result = run_command(
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=42949672960,max "
      "EMEL_MEMORY_TEST_CURRENT_MAX=42949672960 "
      "EMEL_MEMORY_TEST_CURRENT_SWAP=0 EMEL_MEMORY_TEST_CORES=64 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status == 0);
  CHECK(result.output.find("effective_total_bytes=42949672960") !=
        std::string::npos);
  CHECK(result.output.find("cap_bytes=21474836480") != std::string::npos);
}

TEST_CASE("memory overrides validate and build jobs clamp to the cap") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_override_test.txt";
  const auto script = repo_root() / "scripts" / "build_jobs.sh";
  const std::string base =
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=max "
      "EMEL_MEMORY_TEST_CURRENT_MAX=max EMEL_MEMORY_TEST_CURRENT_SWAP=max "
      "EMEL_MEMORY_TEST_CORES=64 ";

  command_result result = run_command(
      base + "EMEL_MEMORY_CAP_PERCENT=51 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("EMEL_MEMORY_CAP_PERCENT must be 1..50") !=
        std::string::npos);

  result = run_command(base + "EMEL_BUILD_JOBS=40 bash " +
                           shell_quote(script.string()) +
                           " --memory-cap-check",
                       output);
  CHECK(result.status == 0);
  CHECK(result.output.find("safe_build_jobs=7") != std::string::npos);
  CHECK(result.output.find("build_jobs=7") != std::string::npos);
  CHECK(result.output.find("clamping EMEL_BUILD_JOBS=40") !=
        std::string::npos);

  result = run_command(base + "EMEL_BUILD_JOBS=invalid bash " +
                           shell_quote(script.string()) +
                           " --memory-cap-check",
                       output);
  CHECK(result.status != 0);
  CHECK(result.output.find("EMEL_BUILD_JOBS must be a positive integer") !=
        std::string::npos);
}

TEST_CASE("memory envelope constructs Linux scope and fails closed on Darwin") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_envelope_command_test.txt";
  const auto script = repo_root() / "scripts" / "build_jobs.sh";
  const std::string base =
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=max "
      "EMEL_MEMORY_TEST_CURRENT_MAX=max EMEL_MEMORY_TEST_CURRENT_SWAP=max "
      "EMEL_MEMORY_TEST_CORES=64 ";

  command_result result = run_command(
      base + "EMEL_MEMORY_TEST_OS=Linux bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status == 0);
  CHECK(result.output.find("systemd-run --user --scope --quiet --same-dir") !=
        std::string::npos);
  CHECK(result.output.find("--property=MemoryMax=53687091200") !=
        std::string::npos);
  CHECK(result.output.find("--property=MemorySwapMax=0") !=
        std::string::npos);

  result = run_command(base + "EMEL_MEMORY_TEST_OS=Darwin bash " +
                           shell_quote(script.string()) +
                           " --memory-cap-check",
                       output);
  CHECK(result.status != 0);
  CHECK(result.output.find("macOS has no supported native aggregate descendant "
                           "memory controller") != std::string::npos);
  CHECK(result.output.find("sampled watchdogs") != std::string::npos);
}

TEST_CASE("memory envelope marker requires observable cgroup limits") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_envelope_failure_test.txt";
  const auto script = repo_root() / "scripts" / "build_jobs.sh";
  const std::string base =
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_CORES=64 EMEL_MEMORY_TEST_OWNED_SCOPE=1 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=max ";

  command_result result = run_command(
      base + "EMEL_MEMORY_TEST_CURRENT_MAX=53687091200 "
             "EMEL_MEMORY_TEST_CURRENT_SWAP=0 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status == 0);
  CHECK(result.output.find("cap_bytes=53687091200") != std::string::npos);

  result = run_command(
      base + "EMEL_MEMORY_TEST_CURRENT_MAX=53687091201 "
             "EMEL_MEMORY_TEST_CURRENT_SWAP=0 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("memory.max=53687091201 exceeds recomputed cap") !=
        std::string::npos);

  result = run_command(
      "env EMEL_MEMORY_TEST_PHYSICAL_BYTES=107374182400 "
      "EMEL_MEMORY_TEST_CORES=64 EMEL_MEMORY_TEST_OWNED_SCOPE=1 "
      "EMEL_MEMORY_TEST_ANCESTOR_MAXES=max,42949672960,85899345920 "
      "EMEL_MEMORY_TEST_CURRENT_MAX=21474836481 "
      "EMEL_MEMORY_TEST_CURRENT_SWAP=0 bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("exceeds recomputed cap 21474836480") !=
        std::string::npos);

  result = run_command(
      base + "EMEL_MEMORY_TEST_CURRENT_MAX=53687091200 "
             "EMEL_MEMORY_TEST_CURRENT_SWAP=max bash " +
          shell_quote(script.string()) + " --memory-cap-check",
      output);
  CHECK(result.status != 0);
  CHECK(result.output.find("memory.swap.max=max, required 0") !=
        std::string::npos);
}

TEST_CASE("sourced production entry ignores memory test environment") {
  const auto output = std::filesystem::temp_directory_path() /
                      "emel_memory_source_trust_test.txt";
  const auto helper = repo_root() / "scripts" / "build_jobs.sh";
  const command_result result = run_command(
      "env EMEL_MEMORY_TEST_OS=Darwin EMEL_MEMORY_TEST_PHYSICAL_BYTES=1 "
      "EMEL_MEMORY_TEST_CURRENT_MAX=1 EMEL_MEMORY_TEST_CURRENT_SWAP=0 bash -c " +
          shell_quote("source " + shell_quote(helper.string())),
      output);
  CHECK(result.status == 0);
}
TEST_CASE("every native build or installer enters the hard envelope first") {
  const auto scripts = repo_root() / "scripts";
  for (const auto &entry : std::filesystem::directory_iterator(scripts)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".sh") continue;
    const std::string text = read_file(entry.path());
    std::size_t first_work = std::string::npos;
    for (const std::string needle : {"cmake --build", "zig build", "ninja ",
                                     "pip install", "pip3 install", "uv pip",
                                     "uv sync"}) {
      const std::size_t position = text.find(needle);
      if (position < first_work) first_work = position;
    }
    if (first_work != std::string::npos) {
      CAPTURE(entry.path().filename().string());
      const std::size_t source = text.find("build_jobs.sh");
      CHECK(source != std::string::npos);
      CHECK(source < first_work);
    }
  }
}

TEST_CASE("material setup scripts enter the hard envelope first") {
  const auto scripts = repo_root() / "scripts";
  for (const std::string name : {"setup_diarization_pytorch_ref_env.sh",
                                 "setup_moshi_cpp_reference.sh",
                                 "setup_personaplex_mlx_reference.sh",
                                 "setup_whisper_cpp_reference.sh"}) {
    const std::string text = read_file(scripts / name);
    CAPTURE(name);
    const std::size_t source = text.find("build_jobs.sh");
    CHECK(source != std::string::npos);
    CHECK(source < text.find("python"));
    CHECK(source < text.find("cmake"));
  }
}

TEST_CASE("direct check and lint gates enter the hard envelope first") {
  const auto scripts = repo_root() / "scripts";
  for (const auto &entry : std::filesystem::directory_iterator(scripts)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".sh") continue;
    const std::string name = entry.path().filename().string();
    if (!(name.starts_with("check_") || name.starts_with("lint_"))) continue;
    const std::string text = read_file(entry.path());
    CAPTURE(name);
    const std::size_t source = text.find("build_jobs.sh");
    CHECK(source != std::string::npos);
    CHECK(source < text.find("if "));
    CHECK(source < text.find("rg "));
  }

  for (const std::string name : {"quality_gates.sh", "build_with_zig.sh",
                                 "test_with_coverage.sh",
                                 "test_with_sanitizers.sh", "fuzz_smoke.sh",
                                 "lint_snapshot.sh", "bench.sh",
                                 "generate_docs.sh", "embedded_size.sh"}) {
    const std::string text = read_file(scripts / name);
    CAPTURE(name);
    CHECK(text.find("build_jobs.sh") != std::string::npos);
  }
}


TEST_CASE("profile scripts enter the hard envelope first") {
  const auto scripts = repo_root() / "scripts";
  for (const auto &entry : std::filesystem::directory_iterator(scripts)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".sh") continue;
    const std::string name = entry.path().filename().string();
    if (!name.starts_with("profile_")) continue;
    const std::string text = read_file(entry.path());
    CAPTURE(name);
    const std::size_t source = text.find("build_jobs.sh");
    CHECK(source != std::string::npos);
    CHECK(source < text.find("xctrace"));
    CHECK(source < text.find("bench_runner"));
  }
}
TEST_CASE("removed memory bypasses and sampled watchdog stay absent") {
  const std::string helper =
      read_file(repo_root() / "scripts" / "build_jobs.sh");
  const std::string gates =
      read_file(repo_root() / "scripts" / "quality_gates.sh");
  CHECK(helper.find("EMEL_ALLOW_UNCAPPED_MEMORY") == std::string::npos);
  CHECK(helper.find("DANGEROUS_ALLOW") == std::string::npos);
  CHECK(gates.find("darwin-rss-watchdog") == std::string::npos);
  CHECK(gates.find("process_tree_rss") == std::string::npos);
  CHECK(helper.find("emel_run_delegated_cgroup") == std::string::npos);
  CHECK(helper.find("delegated cgroup") == std::string::npos);
  CHECK(helper.find("verified systemd --user scopes") != std::string::npos);
}


TEST_CASE("CMake presets expose configuration only") {
  const std::string presets = read_file(repo_root() / "CMakePresets.json");
  CHECK(presets.find("\"buildPresets\"") == std::string::npos);
  CHECK(presets.find("\"testPresets\"") == std::string::npos);
  CHECK(presets.find("\"configurePresets\"") != std::string::npos);
}
#endif
