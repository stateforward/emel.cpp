#include "doctest/doctest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

std::filesystem::path repo_root() {
  return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path();
}

std::string read_text_file(const std::filesystem::path & path) {
  std::ifstream stream(path);
  REQUIRE_MESSAGE(stream.good(), "failed to open file: " << path.string());
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void check_contains(const std::string & content, const std::string_view needle) {
  CHECK_MESSAGE(content.find(needle) != std::string::npos,
                "missing expected text: " << needle);
}

}  // namespace

TEST_CASE("maintained TE fixture is documented in tests/models README") {
  const auto readme_path = repo_root() / "tests" / "models" / "README.md";
  REQUIRE(std::filesystem::exists(readme_path));

  const std::string content = read_text_file(readme_path);
  check_contains(content, "## TE-75M-q8_0.gguf");
  check_contains(content, "tests/models/TE-75M-q8_0.gguf");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF/resolve/main/TE-75M-q8_0.gguf");
  check_contains(content, "119710336");
  check_contains(content, "955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae");
  check_contains(content, "gguf.architecture=omniembed");
  check_contains(content, "mdbr-leaf-ir-vocab.txt");
  check_contains(content, "https://huggingface.co/MongoDB/mdbr-leaf-ir");
  check_contains(content, "https://huggingface.co/MongoDB/mdbr-leaf-ir/resolve/main/vocab.txt");
  check_contains(content, "231508");
  check_contains(content, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3");
  check_contains(content, "TE-75M-q5_0.gguf");
  check_contains(content, "tests/models/TE-75M-q5_0.gguf");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF/resolve/main/TE-75M-q5_0.gguf");
  check_contains(content, "111149248");
  check_contains(content, "c63eb0db4fe4364e05d732063b45adb1dddfa206ba53886ed6d9b1b6fe1f9b73");
}

TEST_CASE("TE proof corpus is defined with narrow pairwise anchors") {
  const auto corpus_dir = repo_root() / "tests" / "embeddings" / "fixtures" / "te75m";
  const auto manifest_path = corpus_dir / "README.md";
  const auto red_square_text_path = corpus_dir / "red-square.txt";
  const auto pure_tone_text_path = corpus_dir / "pure-tone-440hz.txt";

  REQUIRE(std::filesystem::exists(manifest_path));
  REQUIRE(std::filesystem::exists(red_square_text_path));
  REQUIRE(std::filesystem::exists(pure_tone_text_path));

  const std::string manifest = read_text_file(manifest_path);
  check_contains(manifest, "red-square");
  check_contains(manifest, "pure-tone-440hz");
  check_contains(manifest, "RGBA `uint8`");
  check_contains(manifest, "width: `32`");
  check_contains(manifest, "height: `32`");
  check_contains(manifest, "sample rate: `16000`");
  check_contains(manifest, "sample count: `4000`");
  check_contains(manifest, "0.2f * sinf");

  CHECK(read_text_file(red_square_text_path) == std::string{"a red square\n"});
  CHECK(read_text_file(pure_tone_text_path) == std::string{"a pure 440 hertz sine tone\n"});
}

TEST_CASE("maintained TE fixture matches locked local size when present") {
  const auto fixture_path = repo_root() / "tests" / "models" / "TE-75M-q8_0.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local TE fixture size check because the maintained fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 119710336ULL);
}

TEST_CASE("maintained TE q5 fixture matches locked local size when present") {
  const auto fixture_path = repo_root() / "tests" / "models" / "TE-75M-q5_0.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local TE q5 fixture size check because the maintained q5 fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 111149248ULL);
}

TEST_CASE("maintained TE tokenizer vocab matches locked local size") {
  const auto vocab_path = repo_root() / "tests" / "models" / "mdbr-leaf-ir-vocab.txt";
  REQUIRE(std::filesystem::exists(vocab_path));
  CHECK(std::filesystem::file_size(vocab_path) == 231508ULL);
}
