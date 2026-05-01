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
  CHECK(script.find("bench_dependency_manifest_requires_full_gate()") != std::string::npos);
  CHECK(script.find("--write-dependency-manifest") != std::string::npos);
  CHECK(script.find("--check-dependency-manifest") != std::string::npos);
  CHECK(script.find("dependency manifest requires full benchmark gate") != std::string::npos);
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
  CHECK(helper.find("bench_full=true") != std::string::npos);
  CHECK(helper.find("add_bench_suite \"$runner\"") != std::string::npos);
  CHECK(helper.find("tools/bench/*|tools/bench/**/*") != std::string::npos);
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
}
