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
