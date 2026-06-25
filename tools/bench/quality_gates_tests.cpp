#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

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

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("quality gates full benchmark branch preserves failure status") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t full_branch_start = script.find("if $bench_full; then");
  REQUIRE(full_branch_start != std::string::npos);

  const std::size_t scoped_branch_end =
      script.find("if [[ ${#bench_suites[@]} -eq 0 ]]; then", full_branch_start);
  REQUIRE(scoped_branch_end != std::string::npos);

  const std::string full_branch =
      script.substr(full_branch_start, scoped_branch_end - full_branch_start);

  CHECK(full_branch.find("run_step_allow_fail bench_snapshot") != std::string::npos);
  CHECK(full_branch.find("status=$?") != std::string::npos);
  CHECK(full_branch.find("return \"$status\"") != std::string::npos);
  CHECK(full_branch.find("return $?") == std::string::npos);
}

TEST_CASE("quality gates exclude nested sml machine headers from coverage source set") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("is_coverage_excluded_src_file()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("changed_files=()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("src/emel/**/*/sm.hpp") != std::string::npos);
}

TEST_CASE("coverage script enforces thresholds on changed executable lines") {
  const std::string script = read_file(repo_root() / "scripts" / "test_with_coverage.sh");

  CHECK(script.find("COVERAGE_CHANGED_LINE_ONLY") != std::string::npos);
  CHECK(script.find("required_tools+=(python3)") != std::string::npos);
  CHECK(script.find("collect_changed_lines()") != std::string::npos);
  CHECK(script.find("enforce_changed_line_coverage()") != std::string::npos);
  CHECK(script.find("--json \"$coverage_json\"") != std::string::npos);
  CHECK(script.find("changed-line coverage:") != std::string::npos);
  CHECK(script.find("--fail-under-line \"$LINE_COVERAGE_MIN\"") != std::string::npos);
  CHECK(script.find("--fail-under-branch \"$BRANCH_COVERAGE_MIN\"") != std::string::npos);
}

TEST_CASE("quality gates consume benchmark dependency manifest conservatively") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("BENCH_DEPENDENCY_MANIFEST_BASELINE") != std::string::npos);
  CHECK(script.find("tools/bench/dependency_manifest.txt") != std::string::npos);
  CHECK(script.find("BENCH_RUNNER_BINARY") != std::string::npos);
  CHECK(script.find("bench_dependency_manifest_apply_changed_files()") != std::string::npos);
  CHECK(script.find("add_all_benchmark_suites_from_manifest()") != std::string::npos);
  CHECK(script.find("bench_dependency_manifest_requires_full_gate()") != std::string::npos);
  CHECK(script.find("--write-dependency-manifest") != std::string::npos);
  CHECK(script.find("--check-dependency-manifest") != std::string::npos);
  CHECK(script.find("dependency manifest requires full benchmark gate") != std::string::npos);
}

TEST_CASE("quality gates consume parity dependency manifest for runner selection") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("PARITY_DEPENDENCY_MANIFEST_BASELINE") != std::string::npos);
  CHECK(script.find("tools/paritychecker/dependency_manifest.txt") != std::string::npos);
  CHECK(script.find("parity_dependency_manifest_apply_changed_files()") != std::string::npos);
  CHECK(script.find("parity_toolchain_file_requires_full_gate()") != std::string::npos);
  CHECK(script.find("select_full_parity_gate \"paritychecker toolchain change path=$file\"") !=
        std::string::npos);
  CHECK(script.find("add_parity_runner \"$runner\" \"manifest path=$path\"") !=
        std::string::npos);
  CHECK(script.find("select_full_parity_gate \"unmatched parity-relevant change path=$file\"") !=
        std::string::npos);
  CHECK(script.find("scripts/paritychecker.sh\" \"${runner_args[@]}\"") != std::string::npos);
}

TEST_CASE("quality gates check parity manifest freshness before deciding skip branch") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t run_start = script.find("run_parity_gate()");
  REQUIRE(run_start != std::string::npos);

  const std::size_t case_branch = script.find("case \"$QUALITY_GATES_PARITY\" in", run_start);
  REQUIRE(case_branch != std::string::npos);

  const std::string pre_case = script.substr(run_start, case_branch - run_start);
  CHECK(pre_case.find("[[ \"$QUALITY_GATES_PARITY\" != \"always\" ]]") != std::string::npos);
  CHECK(pre_case.find("parity_dependency_manifest_requires_full_gate") != std::string::npos);
  CHECK(pre_case.find("select_full_parity_gate \"dependency manifest freshness gap\"") !=
        std::string::npos);
  CHECK(pre_case.find("parity_dependency_manifest_check_needed") == std::string::npos);
}

TEST_CASE("paritychecker script supports selected maintained runners") {
  const std::string script = read_file(repo_root() / "scripts" / "paritychecker.sh");

  CHECK(script.find("--runner=<name>|--mode=<name>") != std::string::npos);
  CHECK(script.find("selected_runners=()") != std::string::npos);
  CHECK(script.find("gbnf)\n      runner=\"gbnf_parser\"") != std::string::npos);
  CHECK(script.find("paritychecker: runner=$runner") != std::string::npos);
  CHECK(script.find("--test-case=\"*tokens across tiny models*\"") != std::string::npos);
  CHECK(script.find("--test-case=\"*gbnf parser outputs*\"") != std::string::npos);
  CHECK(script.find("--test-case=\"*kernel outputs*\"") != std::string::npos);
  CHECK(script.find("--test-case=\"*jinja parser and formatter outputs*\"") !=
        std::string::npos);
  CHECK(script.find("--test-case=\"paritychecker matches current maintained generation "
                    "publication against live reference\"") != std::string::npos);
}

TEST_CASE("quality gates preserve failing lane status in parallel children") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("run_step()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("run_step_allow_fail()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("local status=0") != std::string::npos);
  CHECK(helper.find("if \"$@\"; then") != std::string::npos);
  CHECK(helper.find("status=$?") != std::string::npos);
  CHECK(helper.find("return \"$status\"") != std::string::npos);
}

TEST_CASE("quality gates can run independent heavy lanes in ordered parallel group") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("QUALITY_GATES_PARALLEL") != std::string::npos);
  CHECK(script.find("parallel_enabled()") != std::string::npos);
  CHECK(script.find("start_parallel_step bench_snapshot run_benchmark_gates") !=
        std::string::npos);
  CHECK(script.find("start_parallel_step test_with_coverage run_coverage_gate") !=
        std::string::npos);
  CHECK(script.find("start_parallel_step paritychecker run_parity_gate") != std::string::npos);
  CHECK(script.find("start_parallel_step fuzz_smoke run_fuzz_gate") != std::string::npos);
  CHECK(script.find("quality_gates: log begin name=$name") != std::string::npos);
  CHECK(script.find("quality_gates: log end name=$name status=$status") != std::string::npos);
  CHECK(script.find("EMEL_QUALITY_GATES_PARALLEL_CHILD=1") != std::string::npos);
  CHECK(script.find("set +e\n    \"$@\" >\"$log_file\" 2>&1") != std::string::npos);
  CHECK(script.find("printf '%s\\n' \"$status\" >\"$status_file\"") != std::string::npos);
}

TEST_CASE("quality gate script changes keep mandatory lanes conservative") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");

  const std::size_t infer_start = script.find("infer_quality_gate_scope()");
  REQUIRE(infer_start != std::string::npos);

  const std::size_t infer_end = script.find("if [[ \"$QUALITY_GATES_PARITY\"", infer_start);
  REQUIRE(infer_end != std::string::npos);

  const std::string infer_body = script.substr(infer_start, infer_end - infer_start);
  CHECK(infer_body.find("scripts/quality_gates.sh)") != std::string::npos);
  CHECK(infer_body.find("coverage_all_required=true") != std::string::npos);
  CHECK(infer_body.find("select_full_parity_gate \"quality gate script changed path=$file\"") !=
        std::string::npos);
  CHECK(infer_body.find("quality_gates: select benchmark runner=all "
                        "reason=quality gate script changed path=$file") != std::string::npos);
  CHECK(infer_body.find("bench_all_suites=true") != std::string::npos);
  CHECK(infer_body.find("add_all_benchmark_suites_from_manifest") !=
        std::string::npos);
  CHECK(script.find("\"$QUALITY_GATES_SCOPE\" == \"full\" || "
                    "\"$coverage_all_required\" == \"true\"") != std::string::npos);
}

TEST_CASE("bench script keeps suite-filtered builds out of canonical bench-tools cache") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("bench_suite_build_dir()") != std::string::npos);
  CHECK(script.find("build/bench_tools_ninja_${safe_suite}") != std::string::npos);
  CHECK(script.find("BENCH_COMPARE_BUILD_DIR:-$(bench_suite_build_dir") !=
        std::string::npos);
  CHECK(script.find("BENCH_BUILD_DIR:-$(bench_suite_build_dir") != std::string::npos);
}

TEST_CASE("bench script exposes unfiltered bench tool validation command") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("--test-tools") != std::string::npos);
  CHECK(script.find("--test-tools cannot be combined") != std::string::npos);
  CHECK(script.find("BENCH_TOOLS_TEST_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja") !=
        std::string::npos);
  CHECK(script.find("bench_runner_tests quality_gates_tests") != std::string::npos);
  const std::string ctest_contract =
      "ctest --test-dir \"$build_dir\" -R 'quality_gates_tests|bench_runner_tests'";
  CHECK(script.find(ctest_contract) != std::string::npos);
}

TEST_CASE("benchmark defaults stay bounded for routine quality gates") {
  const std::string quality_gates = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::string bench_runner = read_file(repo_root() / "tools" / "bench" / "bench_runner.cpp");

  CHECK(quality_gates.find("QUALITY_GATES_BENCH_ITERS=\"${EMEL_QUALITY_GATES_BENCH_ITERS:-100}\"") !=
        std::string::npos);
  CHECK(quality_gates.find("QUALITY_GATES_BENCH_RUNS=\"${EMEL_QUALITY_GATES_BENCH_RUNS:-3}\"") !=
        std::string::npos);
  CHECK(quality_gates.find(
            "QUALITY_GATES_BENCH_WARMUP_ITERS=\"${EMEL_QUALITY_GATES_BENCH_WARMUP_ITERS:-10}\"") !=
        std::string::npos);

  CHECK(bench_runner.find("constexpr std::uint64_t k_default_iterations = 100;") !=
        std::string::npos);
  CHECK(bench_runner.find("constexpr std::size_t k_default_runs = 3;") != std::string::npos);
  CHECK(bench_runner.find("constexpr std::uint64_t k_default_warmup_iterations = 10;") !=
        std::string::npos);
}

TEST_CASE("bench script bounds default generation workload") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");
  const std::string generation_bench =
      read_file(repo_root() / "tools" / "bench" / "generation_bench.cpp");
  const std::string diarization_bench =
      read_file(repo_root() / "tools" / "bench" / "diarization" / "sortformer_bench.cpp");

  CHECK(script.find("DEFAULT_GENERATION_WORKLOAD_ID=") != std::string::npos);
  CHECK(script.find("lfm2_single_user_hello_max_tokens_1_v1") != std::string::npos);
  CHECK(script.find("EMEL_GENERATION_WORKLOAD_ID=\"$generation_workload_id\"") !=
        std::string::npos);
  CHECK(script.find("DEFAULT_DIARIZATION_ITERS=\"${EMEL_BENCH_DEFAULT_DIARIZATION_ITERS:-1}\"") !=
        std::string::npos);
  CHECK(script.find("DEFAULT_DIARIZATION_RUNS=\"${EMEL_BENCH_DEFAULT_DIARIZATION_RUNS:-3}\"") !=
        std::string::npos);
  CHECK(script.find("EMEL_BENCH_DIARIZATION_ITERS=\"$diarization_iters\"") !=
        std::string::npos);
  CHECK(script.find("EMEL_BENCH_DIARIZATION_RUNS=\"$diarization_runs\"") !=
        std::string::npos);
  CHECK(script.find("TOLERANCE=\"${BENCH_TOLERANCE:-0.30}\"") != std::string::npos);
  CHECK(script.find("ABS_TOLERANCE_NS=\"${BENCH_ABS_TOLERANCE_NS:-5000}\"") !=
        std::string::npos);
  CHECK(script.find("curr[name] > relative_limit && curr[name] > absolute_limit") !=
        std::string::npos);
  CHECK(generation_bench.find("filter.empty() || filter == \"all\"") !=
        std::string::npos);
  CHECK(diarization_bench.find("EMEL_BENCH_DIARIZATION_ITERS") != std::string::npos);
  CHECK(diarization_bench.find("EMEL_BENCH_DIARIZATION_RUNS") != std::string::npos);
}

TEST_CASE("bench script merges scoped snapshot updates into the full baseline") {
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
  CHECK(script.find("update_snapshot_baseline \"$BASELINE\" \"$current_snapshot\"") !=
        std::string::npos);
  CHECK(script.find("update_snapshot_baseline \"$BASELINE\" \"$CURRENT\"") !=
        std::string::npos);
}

TEST_CASE("bench script rejects scoped compare baseline updates") {
  const std::string script = read_file(repo_root() / "scripts" / "bench.sh");

  CHECK(script.find("if $COMPARE_UPDATE && [[ -n \"$SUITE_FILTER\" ]]") !=
        std::string::npos);
  CHECK(script.find("--compare-update cannot be combined with --suite or --generation-only") !=
        std::string::npos);
}

TEST_CASE("quality gates map benchmark manifest records to scoped or full benchmark gates") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("bench_dependency_manifest_apply_changed_files()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("collect_changed_files()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("runner=\"\"") != std::string::npos);
  CHECK(helper.find("path=\"\"") != std::string::npos);
  CHECK(helper.find("runner=*)") != std::string::npos);
  CHECK(helper.find("path=*)") != std::string::npos);
  CHECK(helper.find("bench_all_suites=true") != std::string::npos);
  CHECK(helper.find("add_all_benchmark_suites_from_manifest") != std::string::npos);
  CHECK(helper.find("add_bench_suite \"$runner\"") != std::string::npos);
  CHECK(helper.find("tools/bench/*|tools/bench/**/*") != std::string::npos);
}

TEST_CASE("quality gates expand full benchmark scope into all manifest suites") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("add_all_benchmark_suites_from_manifest()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("bench_dependency_manifest_record_matches_file()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("add_benchmark_suite_from_manifest \"$priority_runner\"") !=
        std::string::npos);
  CHECK(helper.find("gbnf_rule_parser") != std::string::npos);
  CHECK(helper.find("kernel_aarch64") != std::string::npos);
  CHECK(helper.find("runner=\"\"") != std::string::npos);
  CHECK(helper.find("runner=*)") != std::string::npos);
  CHECK(helper.find("\"$runner\" == \"all\"") != std::string::npos);
  CHECK(helper.find("bench_suite_supported_for_host \"$runner\"") != std::string::npos);
  CHECK(helper.find("add_bench_suite \"$runner\"") != std::string::npos);

  const std::size_t infer_start = script.find("infer_quality_gate_scope()");
  REQUIRE(infer_start != std::string::npos);
  const std::size_t infer_end = script.find("collect_changed_files", infer_start);
  REQUIRE(infer_end != std::string::npos);
  const std::string full_scope = script.substr(infer_start, infer_end - infer_start);
  CHECK(full_scope.find("bench_all_suites=true") != std::string::npos);
  CHECK(full_scope.find("add_all_benchmark_suites_from_manifest") != std::string::npos);
}

TEST_CASE("quality gates skip host-incompatible benchmark suites during full expansion") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("bench_suite_supported_for_host()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end =
      script.find("add_all_benchmark_suites_from_manifest()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("kernel_x86_64)") != std::string::npos);
  CHECK(helper.find("\"x86_64\"") != std::string::npos);
  CHECK(helper.find("\"amd64\"") != std::string::npos);
  CHECK(helper.find("kernel_aarch64)") != std::string::npos);
  CHECK(helper.find("\"aarch64\"") != std::string::npos);
  CHECK(helper.find("\"arm64\"") != std::string::npos);
  CHECK(helper.find("sm_any)") != std::string::npos);
  CHECK(helper.find("EMEL_BENCH_INTERNAL") != std::string::npos);
}

TEST_CASE("quality gates check benchmark manifest before deciding benchmark branch") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t run_start = script.find("run_benchmark_gates()");
  REQUIRE(run_start != std::string::npos);

  const std::size_t full_branch = script.find("if $bench_full; then", run_start);
  REQUIRE(full_branch != std::string::npos);

  const std::string pre_full = script.substr(run_start, full_branch - run_start);
  CHECK(pre_full.find("bench_dependency_manifest_check_needed") != std::string::npos);
  CHECK(pre_full.find("bench_dependency_manifest_requires_full_gate") != std::string::npos);
  CHECK(pre_full.find("bench_full=true") != std::string::npos);
  CHECK(pre_full.find("if ! $bench_full && $bench_all_suites") != std::string::npos);
}

TEST_CASE("quality gates expand broad benchmark scope without monolithic changed gate") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");
  const std::size_t helper_start = script.find("select_full_benchmark_gate()");
  REQUIRE(helper_start != std::string::npos);

  const std::size_t helper_end = script.find("add_parity_runner()", helper_start);
  REQUIRE(helper_end != std::string::npos);

  const std::string helper = script.substr(helper_start, helper_end - helper_start);
  CHECK(helper.find("\"$QUALITY_GATES_SCOPE\" == \"full\"") != std::string::npos);
  CHECK(helper.find("bench_all_suites=true") != std::string::npos);
  CHECK(helper.find("add_all_benchmark_suites_from_manifest") != std::string::npos);
}

TEST_CASE("quality gates bound scoped generation benchmark workload explicitly") {
  const std::string script = read_file(repo_root() / "scripts" / "quality_gates.sh");

  CHECK(script.find("QUALITY_GATES_DEFAULT_GENERATION_WORKLOAD_ID") != std::string::npos);
  CHECK(script.find("lfm2_single_user_hello_max_tokens_1_v1") != std::string::npos);
  CHECK(script.find("EMEL_BENCH_GENERATION_ITERS") != std::string::npos);
  CHECK(script.find("EMEL_BENCH_GENERATION_RUNS") != std::string::npos);
  CHECK(script.find("EMEL_GENERATION_WORKLOAD_ID=\"$generation_workload_id\"") !=
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
