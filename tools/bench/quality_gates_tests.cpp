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
