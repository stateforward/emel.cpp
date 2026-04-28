#include "doctest/doctest.h"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path repo_root() {
  return std::filesystem::path{__FILE__}
      .parent_path()
      .parent_path()
      .parent_path();
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path);
  REQUIRE_MESSAGE(stream.good(), "failed to open file: " << path.string());
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

void check_contains(const std::string &content, const std::string_view needle) {
  CHECK_MESSAGE(content.find(needle) != std::string::npos,
                "missing expected text: " << needle);
}

void check_no_forbidden_includes(
    const std::filesystem::path &path,
    const std::initializer_list<std::string_view> needles) {
  if (!std::filesystem::exists(path)) {
    return;
  }

  const std::string content = read_text_file(path);
  for (const auto needle : needles) {
    CHECK_MESSAGE(content.find(needle) == std::string::npos,
                  "forbidden whisper.cpp/ggml include found in "
                      << path.string() << ": " << needle);
  }
}

} // namespace

TEST_CASE("maintained TE fixture is documented in tests/models README") {
  const auto readme_path = repo_root() / "tests" / "models" / "README.md";
  REQUIRE(std::filesystem::exists(readme_path));

  const std::string content = read_text_file(readme_path);
  check_contains(content, "## TE-75M-q8_0.gguf");
  check_contains(content, "tests/models/TE-75M-q8_0.gguf");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF/resolve/"
                          "main/TE-75M-q8_0.gguf");
  check_contains(content, "119710336");
  check_contains(
      content,
      "955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae");
  check_contains(content, "gguf.architecture=omniembed");
  check_contains(content, "mdbr-leaf-ir-vocab.txt");
  check_contains(content, "https://huggingface.co/MongoDB/mdbr-leaf-ir");
  check_contains(
      content,
      "https://huggingface.co/MongoDB/mdbr-leaf-ir/resolve/main/vocab.txt");
  check_contains(content, "231508");
  check_contains(
      content,
      "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3");
  check_contains(content, "TE-75M-q5_0.gguf");
  check_contains(content, "tests/models/TE-75M-q5_0.gguf");
  check_contains(content, "https://huggingface.co/augmem/TE-75M-GGUF/resolve/"
                          "main/TE-75M-q5_0.gguf");
  check_contains(content, "111149248");
  check_contains(
      content,
      "c63eb0db4fe4364e05d732063b45adb1dddfa206ba53886ed6d9b1b6fe1f9b73");
}

TEST_CASE("maintained Whisper tiny q80 fixture is documented in tests/models "
          "README") {
  const auto readme_path = repo_root() / "tests" / "models" / "README.md";
  REQUIRE(std::filesystem::exists(readme_path));

  const std::string content = read_text_file(readme_path);
  check_contains(content, "## model-tiny-q80.gguf");
  check_contains(content, "tests/models/model-tiny-q80.gguf");
  check_contains(content, "https://huggingface.co/oxide-lab/whisper-tiny-GGUF");
  check_contains(
      content, "https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/"
               "94468a6c81edab8c594d9b1d06ea1dfb64292327/model-tiny-q80.gguf");
  check_contains(content, "40700160");
  check_contains(
      content,
      "52deb0fdcbb9c36b4d570e35f5a65a5ad4275ccdb85e7a06e81a8b05b3743c9d");
  check_contains(content, "general.architecture=whisper");
  check_contains(content, "whisper.n_mels=80");
  check_contains(content, "whisper.n_vocab=51865");
  check_contains(content, "Variant-family scope note");
  check_contains(content, "every variant is already runnable");
  check_contains(content, "loader/model-contract validation only");
  check_contains(content, "not proof that EMEL Whisper ASR runtime");
  check_contains(content, "whisper.cpp/whisper-tiny-q4_k.gguf");
  check_contains(content, "lmgg");
  // Phase 95 narrowing: only q8_0/q4_0/q4_1 belong to the maintained v1.16 family.
  check_contains(content, "narrowed to the three upstream EMEL-loadable Candle-style "
                          "GGUFs");
  check_contains(content, "deferred to a future approved EMEL-owned");
}

TEST_CASE("maintained Whisper tiny q4_0 fixture is documented in tests/models "
          "README") {
  const auto readme_path = repo_root() / "tests" / "models" / "README.md";
  REQUIRE(std::filesystem::exists(readme_path));

  const std::string content = read_text_file(readme_path);
  check_contains(content, "## whisper-tiny-q4_0.gguf");
  check_contains(content, "tests/models/whisper-tiny-q4_0.gguf");
  check_contains(content, "https://huggingface.co/oxide-lab/whisper-tiny-GGUF");
  check_contains(
      content,
      "https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/"
      "94468a6c81edab8c594d9b1d06ea1dfb64292327/whisper-tiny-q4_0.gguf");
  check_contains(content, "22087104");
  check_contains(
      content,
      "b2be6457e86d2c917d0c0eecef8e041ed03c60f64fc5744e6720adfb5141c21b");
  check_contains(content, "loader/model-contract validation only for the `q4_0`");
}

TEST_CASE("maintained Whisper tiny q4_1 fixture is documented in tests/models "
          "README") {
  const auto readme_path = repo_root() / "tests" / "models" / "README.md";
  REQUIRE(std::filesystem::exists(readme_path));

  const std::string content = read_text_file(readme_path);
  check_contains(content, "## whisper-tiny-q4_1.gguf");
  check_contains(content, "tests/models/whisper-tiny-q4_1.gguf");
  check_contains(content, "https://huggingface.co/oxide-lab/whisper-tiny-GGUF");
  check_contains(
      content,
      "https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/"
      "94468a6c81edab8c594d9b1d06ea1dfb64292327/whisper-tiny-q4_1.gguf");
  check_contains(content, "24414464");
  check_contains(
      content,
      "7d40a062a67abeb53784edd326610035089164c9c261cbcfa628e017a07e7a3a");
  check_contains(content, "loader/model-contract validation only for the `q4_1`");
}

TEST_CASE("TE proof corpus is defined with narrow pairwise anchors") {
  const auto corpus_dir =
      repo_root() / "tests" / "embeddings" / "fixtures" / "te75m";
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
  CHECK(read_text_file(pure_tone_text_path) ==
        std::string{"a pure 440 hertz sine tone\n"});
}

TEST_CASE("maintained TE fixture matches locked local size when present") {
  const auto fixture_path =
      repo_root() / "tests" / "models" / "TE-75M-q8_0.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local TE fixture size check because the maintained "
            "fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 119710336ULL);
}

TEST_CASE("maintained TE q5 fixture matches locked local size when present") {
  const auto fixture_path =
      repo_root() / "tests" / "models" / "TE-75M-q5_0.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local TE q5 fixture size check because the maintained q5 "
            "fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 111149248ULL);
}

TEST_CASE("maintained Whisper tiny q80 fixture matches locked local size when "
          "present") {
  const auto fixture_path =
      repo_root() / "tests" / "models" / "model-tiny-q80.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local Whisper tiny fixture size check because the "
            "fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 40700160ULL);
}

TEST_CASE("maintained Whisper tiny q4_0 fixture matches locked local size when "
          "present") {
  const auto fixture_path =
      repo_root() / "tests" / "models" / "whisper-tiny-q4_0.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local Whisper tiny q4_0 fixture size check because the "
            "fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 22087104ULL);
}

TEST_CASE("maintained Whisper tiny q4_1 fixture matches locked local size when "
          "present") {
  const auto fixture_path =
      repo_root() / "tests" / "models" / "whisper-tiny-q4_1.gguf";
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping local Whisper tiny q4_1 fixture size check because the "
            "fixture is not present");
    return;
  }

  CHECK(std::filesystem::file_size(fixture_path) == 24414464ULL);
}

TEST_CASE("maintained TE tokenizer vocab matches locked local size") {
  const auto vocab_path =
      repo_root() / "tests" / "models" / "mdbr-leaf-ir-vocab.txt";
  REQUIRE(std::filesystem::exists(vocab_path));
  CHECK(std::filesystem::file_size(vocab_path) == 231508ULL);
}

TEST_CASE("EMEL Whisper sources reference no whisper.cpp or ggml headers") {
  // Build the forbidden include strings via concatenation so this test source
  // file does not itself contain the literal include substrings that the
  // scanner below searches for. Spelling them inline would self-trip the test.
  const std::string ws_dot_h = std::string{"<"} + "whisper" + ".h>";
  const std::string ws_q = std::string{"\""} + "whisper" + ".h\"";
  const std::string ws_bare = std::string{"<"} + "whisper" + ">";
  const std::string gg_dot_h = std::string{"<"} + "ggml" + ".h>";
  const std::string gg_q = std::string{"\""} + "ggml" + ".h\"";
  const std::string gg_bare = std::string{"<"} + "ggml" + ">";
  const std::string ws_cpp = std::string{"<"} + "whisper-cpp" + ">";
  const std::initializer_list<std::string_view> forbidden = {
      ws_dot_h, ws_q, ws_bare, gg_dot_h, gg_q, gg_bare, ws_cpp,
  };

  // EMEL-owned Whisper source must not bootstrap from whisper.cpp/ggml objects.
  const auto whisper_dir = repo_root() / "src" / "emel" / "model" / "whisper";
  REQUIRE(std::filesystem::exists(whisper_dir));
  REQUIRE(std::filesystem::is_directory(whisper_dir));

  std::vector<std::filesystem::path> scanned = {};
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(whisper_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto extension = entry.path().extension().string();
    if (extension != ".hpp" && extension != ".cpp" && extension != ".h" &&
        extension != ".cc") {
      continue;
    }
    check_no_forbidden_includes(entry.path(), forbidden);
    scanned.push_back(entry.path());
  }
  CHECK_MESSAGE(!scanned.empty(),
                "expected at least one EMEL whisper source file under "
                    << whisper_dir.string());

  // The Whisper loader/contract test surface must also stay isolated from any
  // whisper.cpp/ggml header so the parity lane cannot leak into EMEL state.
  check_no_forbidden_includes(
      repo_root() / "tests" / "model" / "loader" / "lifecycle_tests.cpp",
      forbidden);
  check_no_forbidden_includes(
      repo_root() / "tests" / "model" / "fixture_manifest_tests.cpp",
      forbidden);
}
