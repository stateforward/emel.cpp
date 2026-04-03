#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <doctest/doctest.h>

#include "../../tools/generation_fixture_registry.hpp"

namespace {

std::filesystem::path repo_root() {
  std::filesystem::path path = std::filesystem::path(__FILE__);
  if (path.is_relative()) {
    path = std::filesystem::current_path() / path;
  }
  return path.parent_path().parent_path().parent_path();
}

std::string read_file(const std::filesystem::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

}  // namespace

TEST_CASE("maintained fixture registry freezes Bonsai provenance and executable truth") {
  using emel::tools::generation_fixture_registry::k_bonsai_generation_fixture;
  using emel::tools::generation_fixture_registry::k_generation_parity_contract_live_reference_generation;

  CHECK(k_bonsai_generation_fixture.name == "Bonsai-1.7B.gguf");
  CHECK(k_bonsai_generation_fixture.slug == "bonsai_1_7b");
  CHECK(k_bonsai_generation_fixture.fixture_rel == "tests/models/Bonsai-1.7B.gguf");
  CHECK(k_bonsai_generation_fixture.reference_engine == "llama.cpp");
  CHECK(k_bonsai_generation_fixture.reference_repository ==
        "https://github.com/PrismML-Eng/llama.cpp.git");
  CHECK(k_bonsai_generation_fixture.reference_ref ==
        "f5dda7207ed5837f1c83c2f52f851ad9b933d2fd");
  CHECK(k_bonsai_generation_fixture.generation_parity_contract ==
        k_generation_parity_contract_live_reference_generation);
  CHECK(k_bonsai_generation_fixture.source_repo ==
        "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf");
  CHECK(k_bonsai_generation_fixture.download_url ==
        "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf");
  CHECK(k_bonsai_generation_fixture.sha256 ==
        "0ae245fc08236af7cb64caff164937e53a8d54af611b0f398cc992c0a5ba70c4");
  CHECK(k_bonsai_generation_fixture.architecture == "qwen3");
  CHECK(k_bonsai_generation_fixture.tokenizer_model == "gpt2");
  CHECK(k_bonsai_generation_fixture.tokenizer_pre == "qwen2");
  CHECK(k_bonsai_generation_fixture.weight_format == "Q1_0_g128");
  CHECK(k_bonsai_generation_fixture.size_bytes == 248302272u);
  CHECK(k_bonsai_generation_fixture.context_length == 32768);
  CHECK(k_bonsai_generation_fixture.block_count == 28);
  CHECK(k_bonsai_generation_fixture.embedding_length == 2048);
  CHECK(k_bonsai_generation_fixture.attention_head_count == 16);
  CHECK(k_bonsai_generation_fixture.attention_head_count_kv == 8);
  CHECK(k_bonsai_generation_fixture.generation_supported);
  CHECK_FALSE(k_bonsai_generation_fixture.current_publication);

  const std::filesystem::path fixture_path = repo_root() / k_bonsai_generation_fixture.fixture_rel;
  REQUIRE(std::filesystem::exists(fixture_path));
  CHECK(std::filesystem::file_size(fixture_path) == k_bonsai_generation_fixture.size_bytes);
}

TEST_CASE("generation fixture sets separate parity support from benchmark publication scope") {
  bool saw_bonsai = false;
  bool saw_bonsai_bench = false;
  bool saw_qwen3 = false;
  bool saw_lfm2 = false;
  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures) {
    saw_bonsai = saw_bonsai || fixture.name == "Bonsai-1.7B.gguf";
    saw_qwen3 = saw_qwen3 || fixture.name == "Qwen3-0.6B-Q8_0.gguf";
    saw_lfm2 = saw_lfm2 || fixture.name == "LFM2.5-1.2B-Thinking-Q4_K_M.gguf";
    CHECK(fixture.generation_supported);
  }

  for (const auto & fixture :
       emel::tools::generation_fixture_registry::k_supported_generation_bench_fixtures) {
    saw_bonsai_bench = saw_bonsai_bench || fixture.name == "Bonsai-1.7B.gguf";
  }

  CHECK(saw_bonsai);
  CHECK_FALSE(saw_bonsai_bench);
  CHECK(saw_qwen3);
  CHECK(saw_lfm2);
}

TEST_CASE("generation fixture registry models reference identity per fixture with optional commit pin") {
  using emel::tools::generation_fixture_registry::k_generation_parity_contract_append_only_baseline;
  using emel::tools::generation_fixture_registry::k_generation_parity_contract_live_reference_generation;
  using emel::tools::generation_fixture_registry::k_bonsai_generation_fixture;
  using emel::tools::generation_fixture_registry::k_lfm2_generation_fixture;
  using emel::tools::generation_fixture_registry::k_qwen3_generation_fixture;
  using emel::tools::generation_fixture_registry::k_supported_generation_parity_fixtures;

  for (const auto & fixture : k_supported_generation_parity_fixtures) {
    CHECK_FALSE(fixture.reference_engine.empty());
    CHECK_FALSE(fixture.reference_repository.empty());
  }

  CHECK(k_qwen3_generation_fixture.reference_ref.empty());
  CHECK(k_lfm2_generation_fixture.reference_ref.empty());
  CHECK_FALSE(k_bonsai_generation_fixture.reference_ref.empty());
  CHECK(k_qwen3_generation_fixture.generation_parity_contract ==
        k_generation_parity_contract_live_reference_generation);
  CHECK(k_lfm2_generation_fixture.generation_parity_contract ==
        k_generation_parity_contract_append_only_baseline);
  CHECK(k_bonsai_generation_fixture.generation_parity_contract ==
        k_generation_parity_contract_live_reference_generation);
}

TEST_CASE("Bonsai model ledger distinguishes the fixture filename from the weight format") {
  const std::string readme = read_file(repo_root() / "tests/models/README.md");

  REQUIRE_FALSE(readme.empty());
  CHECK(readme.find("Bonsai-1.7B.gguf") != std::string::npos);
  CHECK(readme.find("Q1_0_g128") != std::string::npos);
  CHECK(readme.find("Bonsai-1.7B-Q1_0_g128.gguf") != std::string::npos);
}
